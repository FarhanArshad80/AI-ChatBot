import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

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

# 4. System Prompt
PLM_SYSTEM_PROMPT = (
    "You are an expert Product Lifecycle Management (PLM) Analyst. "
    "Ensure the output is clean, professional plain text. "
    "\nAnalyze these stages: "
    "1. RESEARCH AND DEVELOPMENT: Core innovation and problem solved. "
    "2. MANUFACTURING ACTIVITIES: Materials and production methods. "
    "3. FINANCIAL ACTIVITIES: Price point and target market. "
    "4. EFFECTIVE INFORMATION SYSTEM: Data and feedback loops. "
    "5. MARKETING AND PROMOTION: Branding and selling points. "
    "6. PRODUCT EVOLUTION: Suggested future improvements."
)


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

        # Update in-memory history
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=clean_response)]))

        # Save to Supabase
        supabase.table("chat_history").insert({
            "user_message": user_input,
            "bot_response": clean_response,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        return jsonify({"reply": clean_response})

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


@app.route("/admin/history", methods=["GET"])
def admin_history():
    """Admin endpoint to fetch all chat history from Supabase"""
    try:
        response = supabase.table("chat_history") \
            .select("*") \
            .order("created_at", desc=True) \
            .execute()
        return jsonify({"history": response.data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/admin/history/<int:record_id>", methods=["DELETE"])
def delete_history(record_id):
    """Delete a specific chat record"""
    try:
        supabase.table("chat_history").delete().eq("id", record_id).execute()
        return jsonify({"status": "Deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)