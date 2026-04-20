import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_id = "gemini-3-flash-preview"  # use stable model for API serving

# Store history in memory (per server session)
history = []

@app.route("/chat", methods=["POST"])
def chat():
    global history

    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    # Build contents exactly like your original code
    current_contents = history + [
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    ]

    full_response = ""

    try:
        # Stream and collect response (same logic as your code)
        print(client)
        for chunk in client.models.generate_content_stream(
            model=model_id, contents=current_contents
        ):
            if chunk.text:
                full_response += chunk.text

        # Update history exactly like your original code
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=full_response)]))

        return jsonify({"reply": full_response})

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

if __name__ == "__main__":
    app.run(debug=True, port=5000)