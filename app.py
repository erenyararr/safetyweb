from flask import Flask, render_template_string, request, redirect, url_for, session, send_file
import fitz  # PyMuPDF
import openai
import datetime
import os
import io
import uuid
import psycopg2
import psycopg2.extras
import numpy as np
import re

# --- API key (openai==0.28.1 interface) ---
from config import API_KEY
openai.api_key = API_KEY

# User login (hardcoded)
USERNAME = "selectsafety"
PASSWORD = "eren1234"

# Postgres connection (Railway provides DATABASE_URL)
DB_URL = os.getenv("DATABASE_URL")

# Ephemeral PDF storage
PDF_STORE = {}  # {file_id: (filename, bytes)}

# ======================
# DB INIT
# ======================
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
        embedding FLOAT8[],
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()

# ======================
# Helper Functions
# ======================

def extract_text_from_pdf(pdf_file):
    """Extract all text from PDF"""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def get_embedding(text: str):
    emb = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return emb["data"][0]["embedding"]

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

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

def analyze_with_gpt(text, method="Five Whys", out_lang="English", feedback=None, similar_cases=None):
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": build_prompt(text, method, out_lang, feedback, similar_cases)}],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()

# ---------- Styled PDF Generator ----------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors

def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(colors.HexColor("#06b6d4"))
    canvas.drawString(doc.leftMargin, doc.height + doc.topMargin - 10, "✈ AI Safety Report Analyzer")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(doc.leftMargin + doc.width, 12, f"Page {doc.page}")
    canvas.restoreState()

def generate_pdf(current_report, similar_cases=None, title="Safety Report"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm, topMargin=20*mm, bottomMargin=16*mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], alignment=TA_CENTER,
        fontName="Helvetica-Bold", fontSize=18, textColor=colors.HexColor("#22d3ee"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
        fontSize=14, textColor=colors.HexColor("#38bdf8"))
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica",
        fontSize=10.5, leading=14.5)

    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#0ea5e9"), spaceAfter=8))

    elements.append(Paragraph("<b>Current Report:</b>", h2))
    elements.append(Paragraph(current_report.replace("\n","<br/>"), body))
    elements.append(Spacer(1, 12))

    if similar_cases:
        elements.append(Paragraph("<b>Similar Cases:</b>", h2))
        for idx, case in enumerate(similar_cases, 1):
            elements.append(Paragraph(f"<b>Case {idx}:</b>", h2))
            elements.append(Paragraph(case.replace("\n","<br/>"), body))
            elements.append(Spacer(1, 10))

    doc.build(elements, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buf.seek(0)
    return buf.read()

# ======================
# HTML Page
# ======================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

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
      <h1 class="text-4xl font-extrabold bg-gradient-to-r from-cyan-400 via-emerald-400 to-blue-400 bg-clip-text text-transparent drop-shadow-lg">
        ✈️ AI Safety Report Analyzer
      </h1>

      {% if session.get('logged_in') %}
        <form action="{{ url_for('logout') }}" method="post">
          <button class="text-sm bg-rose-500/90 hover:bg-rose-500 px-3 py-2 rounded-lg">Logout</button>
        </form>
      {% endif %}
    </div>

    {% if not session.get('logged_in') %}
      <!-- LOGIN CARD -->
      <div class="max-w-md mx-auto backdrop-blur-lg bg-white/10 border border-white/20 shadow-2xl p-8 rounded-3xl">
        <h2 class="text-2xl font-semibold text-center mb-6">Login</h2>
        <form method="post" action="{{ url_for('login') }}" class="space-y-4">
          <input name="username" placeholder="Username" class="w-full p-3 rounded-lg bg-slate-800/70 border border-slate-700" required>
          <input name="password" type="password" placeholder="Password" class="w-full p-3 rounded-lg bg-slate-800/70 border border-slate-700" required>
          <button class="w-full py-3 rounded-xl font-bold text-white bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 hover:scale-105 transition transform shadow-lg">
            Sign In
          </button>
        </form>
      </div>
    {% else %}

    <p class="text-slate-400 mb-10">Analyze aviation safety reports with AI assistance</p>

    <!-- Upload Form -->
    <form hx-post="{{ url_for('analyze') }}" hx-target="#reports" hx-swap="beforeend" enctype="multipart/form-data"
          class="backdrop-blur-lg bg-white/10 border border-white/20 shadow-2xl p-8 rounded-3xl mb-12 space-y-6">
      
      <div>
        <label class="block text-lg font-semibold text-cyan-300 mb-2">Upload PDF</label>
        <input type="file" name="pdf" accept=".pdf" required
               class="block w-full text-sm text-slate-300 file:mr-4 file:py-2 file:px-4
                      file:rounded-full file:border-0
                      file:text-sm file:font-semibold
                      file:bg-cyan-500 file:text-white
                      hover:file:bg-cyan-600"/>
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

    {% endif %}
  </div>
</body>
</html>
"""

# ======================
# Flask Routes
# ======================
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
    if not session.get("logged_in"): return "Unauthorized", 401

    pdf_file = request.files["pdf"]
    method   = request.form.get("method","Five Whys")
    lang     = request.form.get("lang","English")
    text     = extract_text_from_pdf(pdf_file)
    emb      = get_embedding(text)

    # Find similar reports
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT id, report_text, result_text, embedding FROM reports")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    similar = []
    for r in rows:
        if r["embedding"]:
            sim = cosine_similarity(emb, r["embedding"])
            if sim > 0.75:
                similar.append(r["result_text"])

    result = analyze_with_gpt(text, method, lang, similar_cases=similar)

    rid = str(uuid.uuid4())
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("INSERT INTO reports (id, method, lang, report_text, result_text, embedding) VALUES (%s,%s,%s,%s,%s,%s);",
        (rid, method, lang, text, result, emb))
    conn.commit(); cur.close(); conn.close()

    pdf_bytes = generate_pdf(result, similar_cases=similar)
    file_id = uuid.uuid4().hex
    PDF_STORE[file_id] = (f"report_{file_id}.pdf", pdf_bytes)
    download_url = url_for("download_pdf", file_id=file_id)

    return f"""
    <div class="bg-slate-800/60 p-6 rounded-2xl">
      <h2 class="text-xl font-bold text-cyan-300 mb-2">📄 Report ({method}, {lang})</h2>
      <a href="{download_url}" class="bg-emerald-500 px-3 py-2 rounded-lg text-white">⬇ Download PDF</a>
      <pre class="whitespace-pre-wrap text-sm">{result}</pre>
      <form hx-post="{url_for('feedback')}" hx-target="#reports" hx-swap="beforeend" class="mt-3">
        <input type="hidden" name="report_id" value="{rid}">
        <textarea name="feedback" class="w-full p-2 bg-slate-800/70 rounded-lg" placeholder="Give feedback..."></textarea>
        <button class="bg-cyan-500 px-3 py-2 rounded-lg text-white mt-2">🔁 Update Report</button>
      </form>
    </div>
    """

@app.route("/feedback", methods=["POST"])
def feedback():
    rid = request.form.get("report_id")
    fb  = request.form.get("feedback","")
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, method, lang FROM reports WHERE id=%s;", (rid,))
    row = cur.fetchone(); cur.close(); conn.close()
    updated = analyze_with_gpt(row["report_text"], row["method"], row["lang"], feedback=fb)
    return f"<div class='bg-slate-700 p-4 rounded-lg mt-2'><b>Updated Report:</b><pre>{updated}</pre></div>"

@app.route("/download/<file_id>")
def download_pdf(file_id):
    if file_id not in PDF_STORE: return "Not found", 404
    filename, data = PDF_STORE[file_id]
    return send_file(io.BytesIO(data), mimetype="application/pdf", as_attachment=True, download_name=filename)

if __name__ == "__main__":
    app.run(debug=True)
