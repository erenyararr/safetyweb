from flask import Flask, render_template_string, request, redirect, url_for, session
import fitz  # PyMuPDF
import openai
import datetime
import os
import uuid
import psycopg2
import psycopg2.extras

# --- API key ---
from config import API_KEY
openai.api_key = API_KEY

# Kullanıcı girişi
USERNAME = "selectsafety"
PASSWORD = "eren1234"

# Postgres bağlantısı (Railway’den gelen env)
DB_URL = os.getenv("DATABASE_URL")

# =============== DB INIT =================
def init_db():
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS reports (
        id UUID PRIMARY KEY,
        method TEXT,
        lang TEXT,
        report_text TEXT,
        result_text TEXT,
        embedding vector(1536),
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()

# ================== HELPERS ===================
def extract_text_from_pdf(pdf_file):
    """PDF içinden tüm metni çıkarır"""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def build_prompt(text, method, out_lang, feedback=None, similar_cases=None):
    lang_map = {
        "English": "Write the full analysis in clear, professional English.",
        "Français": "Rédige toute l’analyse en français professionnel et clair."
    }
    lang_line = lang_map.get(out_lang, lang_map["English"])

    extra = ""
    if similar_cases:
        extra += "\n\n📌 The AI has detected similar past safety reports:\n"
        for case in similar_cases:
            extra += f"- {case[:200]}...\n"

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
{extra}
"""
    if feedback:
        base_prompt += f"\n\n*** Additional Reviewer Feedback to incorporate: ***\n{feedback}\n"

    return base_prompt

def get_embedding(text):
    emb = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return emb["data"][0]["embedding"]

def analyze_with_gpt(text, method="Five Whys", out_lang="English", feedback=None, similar_cases=None):
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": build_prompt(text, method, out_lang, feedback, similar_cases)}],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()

# ================= FLASK ===================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

# HTML (login + app)
PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Safety Analyzer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@2.0.3"></script>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-sky-900 to-slate-800 text-slate-200">
  <div class="max-w-6xl mx-auto py-10 px-6">
    <div class="flex justify-between items-center mb-4">
      <h1 class="text-4xl font-extrabold bg-gradient-to-r from-cyan-400 via-emerald-400 to-blue-400 bg-clip-text text-transparent">✈️ AI Safety Report Analyzer</h1>
      {% if session.get('logged_in') %}
        <form action="{{ url_for('logout') }}" method="post">
          <button class="bg-rose-500 px-3 py-2 rounded-lg">Logout</button>
        </form>
      {% endif %}
    </div>

    {% if not session.get('logged_in') %}
      <div class="max-w-md mx-auto bg-white/10 p-8 rounded-3xl">
        <h2 class="text-2xl text-center mb-6">Login</h2>
        <form method="post" action="{{ url_for('login') }}" class="space-y-4">
          <input name="username" placeholder="Username" class="w-full p-3 rounded-lg bg-slate-800/70">
          <input name="password" type="password" placeholder="Password" class="w-full p-3 rounded-lg bg-slate-800/70">
          <button class="w-full py-3 bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 rounded-xl">Sign In</button>
        </form>
      </div>
    {% else %}
      <p class="text-slate-400 mb-10">Analyze safety reports with AI & see similar past cases</p>

      <form hx-post="{{ url_for('analyze') }}" hx-target="#reports" hx-swap="beforeend" enctype="multipart/form-data"
            class="bg-white/10 p-8 rounded-3xl mb-12">
        <label class="block text-lg font-semibold text-cyan-300 mb-2">Upload PDF</label>
        <input type="file" name="pdf" accept=".pdf" class="mb-4">
        <div class="flex gap-4">
          <select name="method" class="flex-1 p-2 bg-slate-800/70">
            <option>Five Whys</option>
            <option>Fishbone</option>
            <option>Bowtie</option>
          </select>
          <select name="lang" class="flex-1 p-2 bg-slate-800/70">
            <option>English</option>
            <option>Français</option>
          </select>
          <button class="px-6 py-3 bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 rounded-xl">🚀 Run Analysis</button>
        </div>
      </form>

      <div id="reports" class="space-y-10"></div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(PAGE)

@app.route("/login", methods=["POST"])
def login():
    if request.form.get("username") == USERNAME and request.form.get("password") == PASSWORD:
        session["logged_in"] = True
    return redirect(url_for("index"))

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("index"))

@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    pdf_file = request.files["pdf"]
    method = request.form.get("method","Five Whys")
    lang = request.form.get("lang","English")
    text = extract_text_from_pdf(pdf_file)

    # Similar past reports bul
    emb = get_embedding(text)
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, result_text, 1 - (embedding <=> %s) as sim FROM reports ORDER BY sim DESC LIMIT 3;", (emb,))
    rows = cur.fetchall()
    similar = [r["report_text"] for r in rows if r["sim"] > 0.75]
    cur.close()
    conn.close()

    result = analyze_with_gpt(text, method, lang, similar_cases=similar)

    # Save into DB
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("INSERT INTO reports (id, method, lang, report_text, result_text, embedding) VALUES (%s,%s,%s,%s,%s,%s);",
                (str(uuid.uuid4()), method, lang, text, result, emb))
    conn.commit()
    cur.close()
    conn.close()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"""
    <div class="bg-slate-800/60 p-6 rounded-2xl">
      <div class="flex justify-between mb-3">
        <h2 class="text-xl font-bold text-cyan-300">📄 Report ({method}, {lang})</h2>
        <span class="text-xs">{now}</span>
      </div>
      <pre class="whitespace-pre-wrap text-sm">{result}</pre>
    </div>
    """
    return block

if __name__ == "__main__":
    app.run(debug=True)
