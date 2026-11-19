# PTHYONGEMNI Web Application

A Flask-based web application featuring an admin panel, dynamic HTML views, environment-based configuration, and virtual environment setup and realtime chat with the modals.

---

## 📁 Project Structure

```text
PTHYONGEMNI/
├── .agents/             # Agent tools or automation scripts
├── .vibe/               # Project configuration/meta files
├── templates/           # Jinja2 / HTML template files
├── venv/                # Python virtual environment
├── .env                 # Environment variables (secrets)
├── .env.example         # Template for .env
├── .gitignore           # Git ignore rules
├── admin.html           # Admin dashboard view
├── app.py               # Main Flask application server
├── index.html           # Main user landing page / dashboard
├── requirements.txt     # Python dependencies
└── skills-lock.json     # Lock file for project skills/tooling
```

---

## 🚀 Getting Started

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy the example environment file and fill in your keys:

   ```bash
   cp .env.example .env
   ```

   | Variable         | What it is                          |
   | ---------------- | ----------------------------------- |
   | `GEMINI_API_KEY` | Google Gemini API key               |
   | `SUPABASE_URL`   | Your Supabase project URL           |
   | `SUPABASE_KEY`   | Your Supabase API key               |
   | `FLASK_DEBUG`    | `1` to enable debug mode (optional) |

4. Start the server (it listens on port 5000):

   ```bash
   python app.py
   ```

   Add `FLASK_DEBUG=1` to `.env` to turn on auto-reload and the debugger
   while developing.

5. Open `index.html` in your browser to chat, or `admin.html` for the admin dashboard.
