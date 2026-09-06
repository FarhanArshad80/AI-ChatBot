import csv
import io
import json
import os
import re
import uuid
from collections import OrderedDict
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timezone

# 1. Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# 2. Initialize the Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_id = "gemini-2.5-flash-lite"

# 3. Initialize Supabase Client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

# Every past turn is replayed to the model on each request, so an unbounded
# history quietly grows the prompt (and its cost) for the whole session.
# Keep a rolling window of the most recent exchanges instead; the full record
# still lives in Supabase.
MAX_HISTORY_TURNS = 12

# One conversation per session rather than one for the whole process. A single
# shared list meant two people using this at once were talking into the same
# thread: each had the other's turns replayed to the model as context, and
# either one's /reset wiped both.
#
# Kept in memory, so it is per-process and does not survive a restart or a
# second worker. That is the right trade for a demo — the durable record is
# in Supabase — but it is the first thing to move if this is ever run with
# more than one worker.
MAX_CONVERSATIONS = 200

conversations = OrderedDict()

# Session ids arrive from the client, so they are checked before being used
# as a dictionary key. An arbitrary string off the wire should not decide
# what the server allocates or how much of it.
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def session_key(raw):
    """The caller's session id, or a fresh one if it is missing or malformed."""
    return raw if isinstance(raw, str) and SESSION_ID_RE.match(raw) else uuid.uuid4().hex


def request_session_key():
    """The session id carried in the request body.

    In the body rather than a header on purpose: a custom header turns every
    call into a CORS preflight, and this is a field the client already sends
    JSON for.

    Read with silent=True because /reset is posted with no body at all, and
    request.json raises a 415 on a request that never claimed to be JSON.
    """
    return session_key((request.get_json(silent=True) or {}).get("session_id"))


def get_history(key):
    """This session's turns, created on first use.

    Reading a conversation moves it to the end, so when the cap is reached it
    is the one nobody has spoken into for longest that goes — not whichever
    happens to have been created first.
    """
    history = conversations.pop(key, [])
    conversations[key] = history

    while len(conversations) > MAX_CONVERSATIONS:
        conversations.popitem(last=False)

    return history


def utc_now_iso():
    """Current UTC time as an ISO 8601 string.

    datetime.utcnow() is deprecated from Python 3.12 and returns a naive
    value, which drops the offset and leaves stored timestamps ambiguous.
    """
    return datetime.now(timezone.utc).isoformat()


# 4. System Prompt
PLM_SYSTEM_PROMPT = (
    "You are an expert Product Lifecycle Management (PLM) Analyst. "
    "Ensure the output is clean, professional plain text and simple"
    "\nAnalyze these stages: "
    "1. RESEARCH AND DEVELOPMENT: Core innovation and problem solved. "
    "2. MANUFACTURING ACTIVITIES: Materials and production methods. "
    "3. FINANCIAL ACTIVITIES: Price point and target market. "
    "4. EFFECTIVE INFORMATION SYSTEM: Data and feedback loops. "
    "5. MARKETING AND PROMOTION: Branding and selling points. "
    "6. PRODUCT EVOLUTION: Suggested future improvements."
)


# ─────────────────────────────────────────────
#  MODEL PLUMBING
# ─────────────────────────────────────────────

def clean_text(text):
    """Strip the markdown emphasis the system prompt already asks it to skip.

    Applied per character, so running it on one streamed piece gives the
    same result as running it on the whole answer at the end.
    """
    return re.sub(r"[*#]", "", text)


def friendly_error(error):
    """Turn a provider exception into something worth showing, plus a status."""
    text = str(error)

    if "429" in text:
        return "Quota reached. Try again later.", 429
    if "404" in text:
        return f"Model '{model_id}' not available.", 404

    return text, 500


def stream_reply(user_input, history):
    """Yield the model's answer in the pieces it arrives in."""
    contents = history + [
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    ]

    for chunk in client.models.generate_content_stream(
        model=model_id,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=PLM_SYSTEM_PROMPT,
            temperature=0.7
        )
    ):
        if chunk.text:
            yield chunk.text


def remember_turn(user_input, reply, key):
    """Record one exchange in this session's memory and in Supabase.

    Trimming in whole pairs keeps user/model roles alternating, which the API
    expects. The model has already answered by the time this runs, so a
    storage outage costs the transcript, not the reply.
    """
    history = get_history(key)

    history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply)]))
    del history[:-2 * MAX_HISTORY_TURNS]

    try:
        supabase.table("chat_history").insert({
            "user_message": user_input,
            "bot_response": reply,
            "created_at": utc_now_iso()
        }).execute()
        return True
    except Exception as storage_error:
        app.logger.warning("Chat not saved to Supabase: %s", storage_error)
        return False


# Every message stays in the rolling window and is replayed to the model on
# each of the next turns, so an enormous one is not paid for once — it is
# paid for MAX_HISTORY_TURNS times, and it crowds out the exchanges that gave
# the conversation its point. Generous for a product description, far short
# of a pasted document.
MAX_MESSAGE_CHARS = 8000


def read_message():
    """The user's message, or the response explaining why there isn't one.

    Returns (text, None) when the request is usable and (None, response) when
    it is not, so both chat routes reject the same things the same way rather
    than each growing its own idea of what a message is.
    """
    # silent=True because request.json raises a bare 415 on a request that
    # never claimed to be JSON, which tells the caller nothing about what to
    # send instead.
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return None, (jsonify({"error": "Send a JSON object with a 'message' field."}), 400)

    raw = body.get("message", "")

    # A number or a list reaches .strip() and raises AttributeError. That was
    # answering a malformed request with a 500 and a stack trace — the shape
    # of answer that says the server broke, when the request did.
    if not isinstance(raw, str):
        return None, (jsonify({"error": "'message' must be a string."}), 400)

    text = raw.strip()

    if not text:
        return None, (jsonify({"error": "Empty message"}), 400)

    if len(text) > MAX_MESSAGE_CHARS:
        return None, (jsonify({
            "error": f"That message is too long — keep it under {MAX_MESSAGE_CHARS:,} characters.",
            "limit": MAX_MESSAGE_CHARS,
            "length": len(text),
        }), 413)

    return text, None


def sse(payload):
    """Frame one server-sent event.

    The blank line is the frame terminator — without it the browser holds the
    event open and nothing arrives until the next one lands.
    """
    return f"data: {json.dumps(payload)}\n\n"


# ─────────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Lightweight probe so the UI can tell whether the backend is reachable."""
    return jsonify({
        "status": "ok",
        "model": model_id,
        "conversations": len(conversations),
        "max_conversations": MAX_CONVERSATIONS,
        "max_turns": MAX_HISTORY_TURNS,
        "max_message_chars": MAX_MESSAGE_CHARS,
        "checked_at": utc_now_iso()
    })


# ─────────────────────────────────────────────
#  CHAT ROUTES
# ─────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    """One request, one whole answer. Kept for callers that cannot stream."""
    user_input, refusal = read_message()
    if refusal:
        return refusal

    key = request_session_key()

    try:
        reply = clean_text("".join(stream_reply(user_input, get_history(key))))
        saved = remember_turn(user_input, reply, key)

        # Echoed back so a caller that arrived without one can adopt the
        # session it was just given instead of starting over on every turn.
        return jsonify({"reply": reply, "saved": saved, "session_id": key})

    except Exception as e:
        message, status = friendly_error(e)
        return jsonify({"error": message}), status


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """The same answer, sent as it is written.

    A six-stage PLM analysis takes long enough that a spinner is all anyone
    sees for most of it. The model already arrives in pieces — this stops
    holding them until the last one lands.

    The message is read here rather than inside the generator: by the time
    Flask starts consuming that generator the request context is gone.
    """
    user_input, refusal = read_message()
    if refusal:
        return refusal

    # Read out here for the same reason the message is: the request context
    # is gone by the time Flask consumes the generator below.
    key = request_session_key()
    history = get_history(key)

    def events():
        pieces = []
        recorded = False
        saved = False

        def record():
            """Record whatever has arrived so far, at most once."""
            nonlocal recorded

            if recorded or not pieces:
                return False

            recorded = True
            return remember_turn(user_input, clean_text("".join(pieces)), key)

        try:
            for piece in stream_reply(user_input, history):
                pieces.append(piece)
                yield sse({"delta": clean_text(piece)})

            saved = record()
        except Exception as e:
            message, _ = friendly_error(e)
            # The status line went out with the first byte, so a failure
            # halfway through has to be reported inside the stream.
            yield sse({"error": message})
            return
        finally:
            # A client that presses stop or navigates away closes this
            # generator at the yield above, raising GeneratorExit — which is
            # not an Exception and so never reaches the handler. Recording
            # here too keeps the conversation the model is replayed identical
            # to the one that was actually on screen; without it the next
            # turn would carry a question with no answer attached to it.
            record()

        yield sse({"done": True, "saved": saved, "session_id": key})

    return Response(
        events(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tells nginx not to sit on the response until it is complete,
            # which would undo the whole point of streaming it.
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/reset", methods=["POST"])
def reset():
    """Clear one session's turns, not everybody's."""
    key = request_session_key()
    conversations.pop(key, None)

    return jsonify({"status": "History cleared", "session_id": key})


# ─────────────────────────────────────────────
#  ADMIN CRUD ROUTES
# ─────────────────────────────────────────────

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 200


def ilike_pattern(term):
    """Wrap a search term as a PostgREST-safe ilike value.

    % and _ are wildcards to LIKE, so a term containing either would match
    far more than it looks like it should. The or_ filter is itself a
    comma-separated list, so a comma in the term would split it into two
    filters. Escaping and quoting makes PostgREST read the whole thing as
    one literal.
    """
    escaped = (
        term.replace("\\", "\\\\")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace('"', '\\"')
    )
    return f'"%{escaped}%"'


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_date(name):
    """A YYYY-MM-DD bound from the query string, or None.

    Anything that is not a plain date is treated as absent rather than
    rejected, which matches how the paging arguments handle junk: a
    malformed filter should widen the view, never narrow it to nothing.
    """
    value = request.args.get(name, "").strip()

    return value if DATE_RE.match(value) else None


def history_query(search, since=None, until=None, count=None):
    """The transcript, newest first, narrowed by search term and date range."""
    query = supabase.table("chat_history").select("*", count=count)

    if search:
        pattern = ilike_pattern(search)
        query = query.or_(
            f"user_message.ilike.{pattern},bot_response.ilike.{pattern}"
        )

    # Both bounds are inclusive of the whole day named. `until` in particular
    # has to reach the end of its date: comparing against the bare date would
    # read as midnight and silently drop everything said on the last day of
    # the range — the day people most often mean when they pick one.
    if since:
        query = query.gte("created_at", f"{since}T00:00:00+00:00")
    if until:
        query = query.lte("created_at", f"{until}T23:59:59.999999+00:00")

    return query.order("created_at", desc=True)


def read_int(name, default, minimum, maximum):
    """A query-string integer, clamped. Junk falls back to the default."""
    try:
        value = int(request.args.get(name, default))
    except (TypeError, ValueError):
        return default

    return max(minimum, min(value, maximum))


@app.route("/admin/history", methods=["GET"])
def admin_history():
    """READ — a page of chat history, newest first.

    This used to hand back the whole table and let the browser search it,
    which is fine for a demo and untenable for a log that keeps growing:
    every page load pulled every record ever written. Searching and paging
    now happen in Postgres, and the response says how many rows the search
    actually matched so the page can say where it is in them.
    """
    try:
        search = request.args.get("q", "").strip()
        since = read_date("from")
        until = read_date("to")
        limit = read_int("limit", DEFAULT_PAGE_SIZE, 1, MAX_PAGE_SIZE)
        offset = read_int("offset", 0, 0, 1_000_000)

        response = history_query(search, since, until, count="exact") \
            .range(offset, offset + limit - 1) \
            .execute()

        return jsonify({
            "history": response.data or [],
            "total": response.count if response.count is not None else len(response.data or []),
            "limit": limit,
            "offset": offset,
            "query": search,
            "from": since,
            "to": until,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/history/export", methods=["GET"])
def export_history():
    """EXPORT — the transcript as a CSV file.

    The admin table is for reading a few records; anything past that —
    handing the log to someone, keeping a copy off Supabase, opening it in a
    spreadsheet — wants a file. ?q=, ?from= and ?to= filter on the same terms
    the admin table uses, so what is on screen is what comes down.
    """
    try:
        search = request.args.get("q", "").strip()
        since = read_date("from")
        until = read_date("to")

        # Filtered by the same query the table uses, so "what is on screen"
        # and "what comes down" cannot drift apart. No range here: an export
        # is meant to be the whole of whatever was asked for.
        response = history_query(search, since, until).execute()
        records = response.data or []

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["id", "created_at", "user_message", "bot_response"])

        # csv handles the quoting. Replies routinely contain commas, quotes
        # and newlines, and hand-joining these fields would tear a single
        # answer across several rows.
        for record in records:
            writer.writerow([
                record.get("id", ""),
                record.get("created_at", ""),
                record.get("user_message", ""),
                record.get("bot_response", ""),
            ])

        # The filename says what is in the file, so a folder of exports can
        # be told apart without opening any of them.
        span = f"{since or 'start'}-to-{until or 'now'}" if (since or until) else \
            f"{datetime.now(timezone.utc):%Y%m%d}"
        filename = f"plm-chat-history-{span}.csv"

        # utf-8-sig: without the BOM, Excel reads a UTF-8 CSV as the local
        # code page and mangles anything outside ASCII.
        return Response(
            buffer.getvalue().encode("utf-8-sig"),
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/history", methods=["POST"])
def create_history():
    """CREATE — Insert a new chat record manually"""
    try:
        body = request.json or {}
        user_message = body.get("user_message", "").strip()
        bot_response  = body.get("bot_response", "").strip()

        if not user_message or not bot_response:
            return jsonify({"error": "user_message and bot_response are required"}), 400

        result = supabase.table("chat_history").insert({
            "user_message": user_message,
            "bot_response": bot_response,
            "created_at": utc_now_iso()
        }).execute()

        return jsonify({"status": "Created", "record": result.data}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/history/<int:record_id>", methods=["PUT"])
def update_history(record_id):
    """UPDATE — Edit an existing chat record"""
    try:
        body = request.json or {}
        updates = {}

        if "user_message" in body:
            updates["user_message"] = body["user_message"].strip()
        if "bot_response" in body:
            updates["bot_response"] = body["bot_response"].strip()

        if not updates:
            return jsonify({"error": "No fields to update"}), 400

        result = supabase.table("chat_history") \
            .update(updates) \
            .eq("id", record_id) \
            .execute()

        return jsonify({"status": "Updated", "record": result.data})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id):
    """DELETE — Remove a specific chat record"""
    try:
        supabase.table("chat_history").delete().eq("id", record_id).execute()
        return jsonify({"status": "Deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/history/all", methods=["DELETE"])
def delete_all_history():
    """DELETE ALL — Remove every chat record (use with caution)"""
    try:
        # Supabase requires a condition; neq('id', 0) matches everything
        supabase.table("chat_history").delete().neq("id", 0).execute()
        return jsonify({"status": "All records deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
