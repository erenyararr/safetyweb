from flask import Flask, render_template_string, request, session, redirect, url_for, make_response
import fitz  # PyMuPDF
import openai
import datetime
import os
from functools import wraps

# --- API key ---
from config import API_KEY
openai.api_key = API_KEY

# --- Simple auth (hard-coded) ---
AUTH_USERNAME = "selectsafety"
AUTH_PASSWORD = "eren1234"

# ======================
# Helper Functions
# ======================

def extract_text_from_pdf(pdf_file):
    """PDF içinden tüm metni çıkarır"""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def build_prompt(text, method, out_lang, feedback=None):
    lang_map = {
        "English": "Write the full analysis in clear, professional English.",
        "Français": "Rédige toute l’analyse en français professionnel et clair."
    }
    lang_line = lang_map.get(out_lang, lang_map["English"])

    base_prompt = f"""
You are an aviation safety analyst AI. Analyze the following safety report using the "{method}" method.

{lang_line}

Return a detailed markdown-formatted analysis with the following structure:

### Incident Summary
- Brief summary of the incident in 2-3 lines.

### Root Cause Analysis ({method})
- Explain the cause(s) of the incident using the selected method.

### Short-term Solution (7 days)
- Actionable recommendations that can be implemented within a week.

### Long-term Solution (30 days)
- Preventative strategies and systemic improvements.

### Severity Level
- Categorize severity as: Minor / Moderate / Major / Critical

Here is the full report text:
{text}
"""
    if feedback:
        base_prompt += f"\n\n*** Additional Reviewer Feedback to incorporate: ***\n{feedback}\n"

    return base_prompt

def analyze_with_gpt(text, method="Five Whys", out_lang="English", feedback=None):
    # openai==0.28.1 arayüzü
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":build_prompt(text, method, out_lang, feedback)}],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()
# ======================
# Flask App
# ======================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "please-change-this-in-production")

# ---- Auth helper ----
def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user"):
            # HTMX istekleri için yönlendirme header'ı
            if request.headers.get("HX-Request"):
                resp = make_response("", 204)
                resp.headers["HX-Redirect"] = url_for("login")
                return resp
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

# ---- Templates ----
TEMPLATE_LOGIN = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Login • Safety Analyzer</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-sky-900 to-slate-800 text-slate-200 flex items-center">
  <div class="max-w-md w-full mx-auto p-6">
    <div class="backdrop-blur-xl bg-white/10 border border-white/20 shadow-2xl p-8 rounded-3xl">
      <div class="text-center mb-6">
        <div class="text-5xl font-extrabold bg-gradient-to-r from-cyan-400 via-emerald-400 to-blue-400 bg-clip-text text-transparent drop-shadow-lg">
          ✈️ AI Safety Report Analyzer
        </div>
        <p class="text-slate-400 mt-3">Please sign in to continue</p>
      </div>

      {% if error %}
      <div class="mb-4 text-sm text-red-300 bg-red-900/30 border border-red-400/30 rounded-lg px-3 py-2">
        {{ error }}
      </div>
      {% endif %}

      <form method="post" class="space-y-4">
        <div>
          <label class="block text-sm mb-1 text-cyan-300">Username</label>
          <input name="username" autocomplete="username" required
                 class="w-full rounded-lg border border-cyan-400/30 bg-slate-900/60 text-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"/>
        </div>
        <div>
          <label class="block text-sm mb-1 text-cyan-300">Password</label>
          <input type="password" name="password" autocomplete="current-password" required
                 class="w-full rounded-lg border border-cyan-400/30 bg-slate-900/60 text-slate-200 px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-400"/>
        </div>
        <button class="w-full px-4 py-3 rounded-xl font-bold text-white bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 hover:scale-[1.02] transition shadow-lg">
          Sign in
        </button>
      </form>
    </div>
  </div>
</body>
</html>
"""

# Ana HTML template (Tailwind + HTMX, tek sayfa, custom file input) — DEĞİŞMEDİ
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Safety Analyzer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@2.0.3"></script>
  <style>
    .file-input-container { position: relative; display: inline-block; width: 100%; }
    .file-input-container input[type=file] {
      position: absolute; left: 0; top: 0; opacity: 0; width: 100%; height: 100%; cursor: pointer;
    }
    .file-input-label {
      display: block; background: #0d6efd; color: white; padding: 10px 16px;
      border-radius: 8px; font-weight: bold; text-align: center; cursor: pointer;
    }
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-sky-900 to-slate-800 text-slate-200">

  <div class="max-w-6xl mx-auto py-10 px-6">
    <!-- HEADER -->
    <h1 class="text-5xl font-extrabold text-center mb-10 bg-gradient-to-r from-cyan-400 via-emerald-400 to-blue-400 bg-clip-text text-transparent drop-shadow-lg">
      ✈️ AI Safety Report Analyzer
    </h1>
    <p class="text-center text-slate-400 mb-10">Analyze aviation safety reports with AI assistance</p>

    <!-- Upload Form -->
    <form hx-post="/analyze" hx-target="#reports" hx-swap="beforeend" enctype="multipart/form-data"
          class="backdrop-blur-lg bg-white/10 border border-white/20 shadow-2xl p-8 rounded-3xl mb-12 space-y-6">

      <div>
        <label class="block text-lg font-semibold text-cyan-300 mb-2">Upload PDF</label>
        <div class="file-input-container">
          <span class="file-input-label">📄 Choose File</span>
          <input type="file" name="pdf" accept=".pdf" required/>
        </div>
      </div>

      <div class="flex flex-wrap gap-4">
        <select name="method" class="flex-1 border rounded-lg p-2 bg-slate-800/70 text-slate-200">
          <option>Five Whys</option>
          <option>Fishbone</option>
          <option>Bowtie</option>
        </select>
        <select name="lang" class="flex-1 border rounded-lg p-2 bg-slate-800/70 text-slate-200">
          <option>English</option>
          <option>Français</option>
        </select>
        <button class="px-6 py-3 rounded-xl font-bold text-white bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 hover:scale-105 transition transform shadow-lg">
          🚀 Run Analysis
        </button>
      </div>
    </form>

    <!-- Reports Output -->
    <div id="reports" class="space-y-10"></div>
  </div>
</body>
</html>
"""
# Health endpoint (Railway/Vercel için faydalı)
@app.route("/health")
def health():
    return "ok", 200

# --- Auth Routes ---
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "").strip()
        if u == AUTH_USERNAME and p == AUTH_PASSWORD:
            session["user"] = u
            return redirect(url_for("index"))
        return render_template_string(TEMPLATE_LOGIN, error="Invalid username or password.")
    # GET
    if session.get("user"):
        return redirect(url_for("index"))
    return render_template_string(TEMPLATE_LOGIN, error=None)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# --- Main App Routes (login zorunlu) ---
@app.route("/")
@login_required
def index():
    return render_template_string(TEMPLATE)

@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    pdf_file = request.files["pdf"]
    method   = request.form.get("method","Five Whys")
    out_lang = request.form.get("lang","English")
    text     = extract_text_from_pdf(pdf_file)

    result = analyze_with_gpt(text, method, out_lang)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"""
    <div class="backdrop-blur-xl bg-slate-800/60 border border-cyan-400/20 shadow-lg p-6 rounded-2xl">
      <div class="flex justify-between items-center mb-3">
        <h2 class="text-xl font-bold text-cyan-300">📄 Report ({method}, {out_lang})</h2>
        <span class="text-xs text-slate-400">{now}</span>
      </div>
      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{result}</pre>

      <!-- Feedback form -->
      <form hx-post="/feedback" hx-target="#reports" hx-swap="beforeend" class="mt-4 space-y-2">
        <input type="hidden" name="method" value="{method}"/>
        <input type="hidden" name="lang" value="{out_lang}"/>
        <textarea name="feedback" rows="3"
                  class="w-full border border-cyan-400/30 rounded-lg p-2 bg-slate-900/60 text-slate-200" placeholder="Give feedback..."></textarea>
        <button class="bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600 text-white px-4 py-2 rounded-xl hover:scale-105 transition">
          🔁 Update Report
        </button>
        <input type="hidden" name="text" value="{text}"/>
      </form>
    </div>
    """
    return block

@app.route("/feedback", methods=["POST"])
@login_required
def feedback():
    method   = request.form.get("method","Five Whys")
    out_lang = request.form.get("lang","English")
    text     = request.form.get("text","")
    fb       = request.form.get("feedback","")

    updated = analyze_with_gpt(text, method, out_lang, feedback=fb)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"""
    <div class="backdrop-blur-xl bg-slate-700/70 border border-emerald-400/30 shadow-lg p-6 rounded-2xl">
      <div class="flex justify-between items-center mb-3">
        <h2 class="text-xl font-bold text-emerald-300">🤖 Updated Report ({method}, {out_lang})</h2>
        <span class="text-xs text-slate-400">{now}</span>
      </div>
      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{updated}</pre>
    </div>
    """
    return block

if __name__ == "__main__":
    app.run(debug=True)
