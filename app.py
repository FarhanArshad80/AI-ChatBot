import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# 2. Initialize the Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model_id = "gemini-3-flash-preview" 

# Store history in memory
history = []

# 3. Updated System Instruction to enforce plain text formatting
PLM_SYSTEM_PROMPT = (
    "You are an expert Product Lifecycle Management (PLM) Analyst. "
   
    " Ensure the output is clean, professional plain text. "
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

    # Combine history with the new user message
    current_contents = history + [
        types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
    ]

    full_response = ""

    try:
        # Generate content
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

        # Secondary cleanup: Ensure no stray Markdown markers escaped the prompt instructions
        clean_response = re.sub(r'[*#]', '', full_response)

        # Update history with the clean version
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_input)]))
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=clean_response)]))

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

if __name__ == "__main__":
    app.run(debug=True, port=5000)