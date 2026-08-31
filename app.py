import os
import re
from flask import Flask, request, jsonify
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
    global history

    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    current_contents = history + [
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    ]

    full_response = ""

    try:
        for chunk in client.models.generate_content_stream(
            model=model_id,
            contents=current_contents,
            config=types.GenerateContentConfig(
                system_instruction=PLM_SYSTEM_PROMPT,
                temperature=0.7
            )
        ):
            if chunk.text:
                full_response += chunk.text

        clean_response = re.sub(r'[*#]', '', full_response)

        # Update in-memory history, dropping the oldest turns once the
        # window is full. Trimming in whole pairs keeps user/model roles
        # alternating, which the API expects.
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=clean_response)]))
        del history[:-2 * MAX_HISTORY_TURNS]

        # Save to Supabase. The model has already answered at this point, so a
        # storage outage should cost the transcript, not the reply — log it
        # and hand the answer back either way.
        try:
            supabase.table("chat_history").insert({
                "user_message": user_input,
                "bot_response": clean_response,
                "created_at": utc_now_iso()
            }).execute()
            saved = True
        except Exception as storage_error:
            app.logger.warning("Chat not saved to Supabase: %s", storage_error)
            saved = False

        return jsonify({"reply": clean_response, "saved": saved})

    except Exception as e:
        if "429" in str(e):
            return jsonify({"error": "Quota reached. Try again later."}), 429
        elif "404" in str(e):
            return jsonify({"error": f"Model '{model_id}' not available."}), 404
        else:
            return jsonify({"error": str(e)}), 500


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
