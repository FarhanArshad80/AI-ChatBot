import csv
import io
import json
import os
import re
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

# Store history in memory
history = []

# Every past turn is replayed to the model on each request, so an unbounded
# history quietly grows the prompt (and its cost) for the whole session.
# Keep a rolling window of the most recent exchanges instead; the full record
# still lives in Supabase.
MAX_HISTORY_TURNS = 12


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


def stream_reply(user_input):
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


def remember_turn(user_input, reply):
    """Record one exchange in memory and in Supabase.

    Trimming in whole pairs keeps user/model roles alternating, which the API
    expects. The model has already answered by the time this runs, so a
    storage outage costs the transcript, not the reply.
    """
    global history

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
        "turns_in_memory": len(history) // 2,
        "max_turns": MAX_HISTORY_TURNS,
        "checked_at": utc_now_iso()
    })


# ─────────────────────────────────────────────
#  CHAT ROUTES
# ─────────────────────────────────────────────

@app.route("/chat", methods=["POST"])
def chat():
    """One request, one whole answer. Kept for callers that cannot stream."""
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    try:
        reply = clean_text("".join(stream_reply(user_input)))
        saved = remember_turn(user_input, reply)

        return jsonify({"reply": reply, "saved": saved})

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
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    def events():
        pieces = []

        try:
            for piece in stream_reply(user_input):
                pieces.append(piece)
                yield sse({"delta": clean_text(piece)})
        except Exception as e:
            message, _ = friendly_error(e)
            # The status line went out with the first byte, so a failure
            # halfway through has to be reported inside the stream.
            yield sse({"error": message})
            return

        saved = remember_turn(user_input, clean_text("".join(pieces)))
        yield sse({"done": True, "saved": saved})

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
    global history
    history = []
    return jsonify({"status": "History cleared"})


# ─────────────────────────────────────────────
#  ADMIN CRUD ROUTES
# ─────────────────────────────────────────────

@app.route("/admin/history", methods=["GET"])
def admin_history():
    """READ — Fetch all chat history from Supabase"""
    try:
        response = supabase.table("chat_history") \
            .select("*") \
            .order("created_at", desc=True) \
            .execute()
        return jsonify({"history": response.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/history/export", methods=["GET"])
def export_history():
    """EXPORT — the transcript as a CSV file.

    The admin table is for reading a few records; anything past that —
    handing the log to someone, keeping a copy off Supabase, opening it in a
    spreadsheet — wants a file. An optional ?q= filters on the same terms
    the search box uses, so what is on screen is what comes down.
    """
    try:
        query = request.args.get("q", "").strip().lower()

        response = supabase.table("chat_history") \
            .select("*") \
            .order("created_at", desc=True) \
            .execute()

        records = response.data or []

        if query:
            records = [
                record for record in records
                if query in (record.get("user_message") or "").lower()
                or query in (record.get("bot_response") or "").lower()
            ]

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

        filename = f"plm-chat-history-{datetime.now(timezone.utc):%Y%m%d}.csv"

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
