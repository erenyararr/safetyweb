from flask import Flask, render_template_string, request, redirect, url_for, session, send_file
import fitz  # PyMuPDF
import openai
import datetime
import os
import io
import uuid
import json
import psycopg2
import psycopg2.extras
import numpy as np
import re

# --- OpenAI API (openai==0.28.1 arayüzü) ---
from config import API_KEY
openai.api_key = API_KEY

# --- Basit login ---
USERNAME = "selectsafety"
PASSWORD = "eren1234"

# --- DB ---
DB_URL = os.getenv("DATABASE_URL")

# ==================== DB INIT ====================
def init_db():
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sreports (
        id UUID PRIMARY KEY,
        method TEXT,
        lang TEXT,
        report_text TEXT,
        result_text TEXT,
        embedding JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()

# ==================== HELPERS ====================
def extract_text_from_pdf(pdf_file):
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
            extra += f"- {case[:250]}...\n"

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
    return emb["data"][0]["embedding"]  # 1536-dim

def cosine_similarity(v1, v2):
    v1 = np.array(v1, dtype=np.float32)
    v2 = np.array(v2, dtype=np.float32)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

def analyze_with_gpt(text, method="Five Whys", out_lang="English", feedback=None, similar_cases=None):
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": build_prompt(text, method, out_lang, feedback, similar_cases)}],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()

# ==================== PDF EXPORT (şık) ====================
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

def _mk_styles():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#22d3ee"),
        spaceAfter=10,
    )
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#38bdf8"),
        spaceBefore=10, spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14.5,
        spaceAfter=6,
    )
    return title_style, h2, body

def generate_pdf(markdown_text, similar_cases=None, title="Safety Report"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm, topMargin=20*mm, bottomMargin=16*mm,
        title=title
    )
    title_style, h2, body = _mk_styles()

    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#0ea5e9"), spaceAfter=8))

    def add_md_block(md_text):
        lines = [ln.rstrip() for ln in md_text.splitlines()]
        paragraph_buffer = []
        bullet_re = re.compile(r"^\s*[-•]\s+")
        numbered_re = re.compile(r"^\s*\d+\.\s+")
        i = 0

        def flush_para():
            nonlocal paragraph_buffer
            if paragraph_buffer:
                p = Paragraph(" ".join(paragraph_buffer).strip(), body)
                elements.append(p)
                paragraph_buffer = []

        while i < len(lines):
            ln = lines[i]
            if not ln.strip():
                flush_para()
                elements.append(Spacer(1, 4)); i += 1; continue
            if ln.startswith("### "):
                flush_para()
                elements.append(Spacer(1, 2))
                elements.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#0ea5e9"), spaceAfter=4))
                elements.append(Paragraph(ln[4:].strip(), h2)); i += 1; continue
            if bullet_re.match(ln):
                flush_para()
                items = []
                while i < len(lines) and bullet_re.match(lines[i]):
                    txt = bullet_re.sub("", lines[i]).strip()
                    items.append(ListItem(Paragraph(txt, body), leftIndent=6)); i += 1
                elements.append(ListFlowable(items, bulletType="bullet", leftPadding=12))
                elements.append(Spacer(1, 4)); continue
            if numbered_re.match(ln):
                flush_para()
                items = []
                while i < len(lines) and numbered_re.match(lines[i]):
                    txt = numbered_re.sub("", lines[i]).strip()
                    items.append(ListItem(Paragraph(txt, body), leftIndent=6)); i += 1
                elements.append(ListFlowable(items, bulletType="1", leftPadding=12))
                elements.append(Spacer(1, 4)); continue
            paragraph_buffer.append(ln); i += 1
        flush_para()

    # Current
    elements.append(Paragraph("Current Report", h2))
    add_md_block(markdown_text)
    elements.append(Spacer(1, 10))

    # Similars
    if similar_cases:
        elements.append(Paragraph("Similar Cases", h2))
        for idx, case in enumerate(similar_cases, 1):
            elements.append(Paragraph(f"Case {idx}", ParagraphStyle("H3", parent=h2, fontSize=12)))
            add_md_block(case)
            elements.append(Spacer(1, 6))

    doc.build(elements, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buf.seek(0)
    return buf

# ==================== FLASK ====================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

# HTML
PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Safety Analyzer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/htmx.org@2.0.3"></script>
  <style>
    button:active { transform: scale(0.97); }
    .btn { transition: transform .08s ease, opacity .15s ease; }
    .btn:hover { opacity: .9; }
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-sky-900 to-slate-800 text-slate-200">

  <div class="max-w-6xl mx-auto py-10 px-6">
    <div class="flex justify-between items-center mb-6">
      <h1 class="text-4xl font-extrabold bg-gradient-to-r from-cyan-400 via-emerald-400 to-blue-400 bg-clip-text text-transparent">
        ✈️ AI Safety Report Analyzer
      </h1>
      {% if session.get('logged_in') %}
        <form action="{{ url_for('logout') }}" method="post">
          <button class="btn bg-rose-500 px-3 py-2 rounded-lg text-white">Logout</button>
        </form>
      {% endif %}
    </div>

    {% if not session.get('logged_in') %}
      <div class="max-w-md mx-auto bg-white/10 p-8 rounded-3xl">
        <h2 class="text-2xl text-center mb-6">Login</h2>
        <form method="post" action="{{ url_for('login') }}" class="space-y-4">
          <input name="username" placeholder="Username" class="w-full p-3 rounded-lg bg-slate-800/70">
          <input name="password" type="password" placeholder="Password" class="w-full p-3 rounded-lg bg-slate-800/70">
          <button class="btn w-full py-3 bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 rounded-xl text-white">Sign In</button>
        </form>
      </div>
    {% else %}
      <p class="text-slate-400 mb-8">Analyze reports with AI, find similar past cases, export elegant PDFs, and iterate with feedback.</p>

      <form hx-post="{{ url_for('analyze') }}" hx-target="#reports" hx-swap="beforeend" enctype="multipart/form-data"
            class="bg-white/10 p-8 rounded-3xl mb-10 space-y-4 border border-white/10">
        <label class="block text-lg font-semibold text-cyan-300">Upload PDF</label>
        <input type="file" name="pdf" accept=".pdf" required class="mb-2 block">
        <div class="flex flex-wrap gap-3">
          <select name="method" class="flex-1 p-2 bg-slate-800/70 rounded-lg">
            <option>Five Whys</option>
            <option>Fishbone</option>
            <option>Bowtie</option>
          </select>
          <select name="lang" class="flex-1 p-2 bg-slate-800/70 rounded-lg">
            <option>English</option>
            <option>Français</option>
          </select>
          <button class="btn px-6 py-3 bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 rounded-xl text-white">🚀 Run Analysis</button>
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

# -------- core: analyze --------
@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    pdf_file = request.files["pdf"]
    method = request.form.get("method","Five Whys")
    lang = request.form.get("lang","English")
    text = extract_text_from_pdf(pdf_file)

    # Embedding çıkar
    emb = get_embedding(text)

    # DB’den tüm raporları çek
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT id, result_text, embedding FROM sreports ORDER BY created_at DESC;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # Benzerlikleri hesapla (JSONB -> list)
    similars = []
    for r in rows:
        if not r["embedding"]:
            continue
        vec = r["embedding"]
        # bazı Postgres sürümlerinde JSONB python dict/list olarak gelir, bazen str olabilir:
        if isinstance(vec, str):
            try:
                vec = json.loads(vec)
            except Exception:
                continue
        try:
            sim = cosine_similarity(emb, vec)
        except Exception:
            continue
        if sim >= 0.75:
            similars.append({"id": str(r["id"]), "text": r["result_text"], "sim": sim})

    # GPT analizi (benzerler sadece içerik olarak gönderiliyor)
    similar_texts = [s["text"] for s in similars]
    result = analyze_with_gpt(text, method, lang, similar_cases=similar_texts)

    # DB’ye kaydet
    rid = str(uuid.uuid4())
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sreports (id, method, lang, report_text, result_text, embedding) VALUES (%s,%s,%s,%s,%s,%s);",
        (rid, method, lang, text, result, json.dumps(emb))
    )
    conn.commit()
    cur.close()
    conn.close()

    # UI bloğu
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # Similar placeholder
    similar_placeholder = f"<div id='sim-{rid}' class='mt-3 text-sm text-slate-400'>No similar cases loaded yet.</div>"
    block = f"""
    <div class="bg-slate-800/60 p-6 rounded-2xl border border-white/10">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
        <h2 class="text-xl font-bold text-cyan-300">📄 Report ({method}, {lang})</h2>
        <span class="text-xs">{now}</span>
      </div>

      <div class="flex flex-wrap gap-2 mb-3">
        <a href="{{{{ url_for('download_current_pdf', report_id='{rid}') }}}}" class="btn bg-emerald-600 px-3 py-2 rounded-lg text-white">⬇ Download PDF (Current)</a>
        <a href="{{{{ url_for('download_with_similars_pdf', report_id='{rid}') }}}}" class="btn bg-teal-600 px-3 py-2 rounded-lg text-white">⬇ Download PDF (+ Similar)</a>
        <button class="btn bg-indigo-600 px-3 py-2 rounded-lg text-white"
                hx-get="{{{{ url_for('similar_cases', report_id='{rid}') }}}}"
                hx-target="#sim-{rid}" hx-swap="outerHTML">
          🔎 Find Similar Cases
        </button>
      </div>

      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{result}</pre>

      <!-- Similar cases placeholder -->
      {similar_placeholder}

      <!-- Feedback form (append new updated card) -->
      <form hx-post="{{{{ url_for('feedback') }}}}" hx-target="#reports" hx-swap="beforeend" class="mt-4 space-y-2">
        <input type="hidden" name="report_id" value="{rid}">
        <textarea name="feedback" rows="3" class="w-full p-2 bg-slate-800/70 rounded-lg" placeholder="Give feedback..."></textarea>
        <button class="btn bg-cyan-500 px-4 py-2 rounded-lg text-white">🔁 Update Report</button>
      </form>
    </div>
    """
    return block

# -------- find similar (buton) --------
@app.route("/similar/<report_id>")
def similar_cases(report_id):
    if not session.get("logged_in"):
        return "Unauthorized", 401

    # raporun embeddingini çek
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT embedding FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    if not row or not row["embedding"]:
        cur.close(); conn.close()
        return f"<div id='sim-{report_id}' class='mt-3 text-sm text-slate-400'>No embedding stored.</div>"

    emb = row["embedding"]
    if isinstance(emb, str):
        try:
            emb = json.loads(emb)
        except Exception:
            cur.close(); conn.close()
            return f"<div id='sim-{report_id}' class='mt-3 text-sm text-rose-400'>Invalid embedding format.</div>"

    # tüm diğer raporlar
    cur.execute("SELECT id, result_text, embedding FROM sreports WHERE id<>%s ORDER BY created_at DESC;", (report_id,))
    rows = cur.fetchall()
    cur.close(); conn.close()

    found = []
    for r in rows:
        vec = r["embedding"]
        if isinstance(vec, str):
            try:
                vec = json.loads(vec)
            except Exception:
                continue
        try:
            sim = cosine_similarity(emb, vec)
        except Exception:
            continue
        if sim >= 0.75:
            found.append({"id": str(r["id"]), "text": r["result_text"], "sim": sim})

    if not found:
        return f"<div id='sim-{report_id}' class='mt-3 text-sm text-slate-400'>No similar cases found.</div>"

    found = sorted(found, key=lambda x: x["sim"], reverse=True)[:5]
    items = "".join(
        f"<li class='mb-2'><span class='text-emerald-300 font-semibold'>sim={f['sim']:.2f}</span> &mdash; {f['text'][:220]}...</li>"
        for f in found
    )
    return f"""
    <div id="sim-{report_id}" class="mt-3 text-sm bg-slate-900/50 p-3 rounded border border-slate-700">
      <div class="font-semibold text-cyan-300 mb-2">Similar Cases</div>
      <ul class="list-disc pl-5">{items}</ul>
    </div>
    """

# -------- feedback (yeni kart olarak ekle) --------
@app.route("/feedback", methods=["POST"])
def feedback():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    rid = request.form.get("report_id", "")
    fb = request.form.get("feedback", "")

    # rapor temel bilgileri
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, method, lang FROM sreports WHERE id=%s;", (rid,))
    row = cur.fetchone()
    cur.close(); conn.close()

    if not row:
        return "<div class='text-rose-400 mt-3'>Report not found.</div>"

    updated = analyze_with_gpt(row["report_text"], row["method"], row["lang"], feedback=fb)

    # Yeni bir kayıt olarak ekleyelim (feedback sonucu ayrı kart istiyorsun)
    new_id = str(uuid.uuid4())
    emb_new = get_embedding(row["report_text"])
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sreports (id, method, lang, report_text, result_text, embedding) VALUES (%s,%s,%s,%s,%s,%s);",
        (new_id, row["method"], row["lang"], row["report_text"], updated, json.dumps(emb_new))
    )
    conn.commit(); cur.close(); conn.close()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    <div class="bg-slate-700/70 p-6 rounded-2xl border border-white/10">
      <div class="flex justify-between mb-2">
        <h3 class="text-emerald-300 font-bold">🤖 Updated Report</h3>
        <span class="text-xs text-slate-400">{now}</span>
      </div>

      <div class="flex flex-wrap gap-2 mb-3">
        <a href="{{{{ url_for('download_current_pdf', report_id='{new_id}') }}}}" class="btn bg-emerald-600 px-3 py-2 rounded-lg text-white">⬇ Download PDF (Current)</a>
        <a href="{{{{ url_for('download_with_similars_pdf', report_id='{new_id}') }}}}" class="btn bg-teal-600 px-3 py-2 rounded-lg text-white">⬇ Download PDF (+ Similar)</a>
        <button class="btn bg-indigo-600 px-3 py-2 rounded-lg text-white"
                hx-get="{{{{ url_for('similar_cases', report_id='{new_id}') }}}}"
                hx-target="#sim-{new_id}" hx-swap="outerHTML">
          🔎 Find Similar Cases
        </button>
      </div>

      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{updated}</pre>
      <div id="sim-{new_id}" class="mt-3 text-sm text-slate-400">No similar cases loaded yet.</div>
    </div>
    """

# -------- PDF: current only --------
@app.route("/download/current/<report_id>")
def download_current_pdf(report_id):
    if not session.get("logged_in"):
        return "Unauthorized", 401

    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT result_text FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    cur.close(); conn.close()

    if not row:
        return "Not found", 404

    pdf = generate_pdf(row["result_text"], similar_cases=None, title="Safety Report")
    return send_file(pdf, as_attachment=True, download_name="report.pdf")

# -------- PDF: current + similars --------
@app.route("/download/with-similars/<report_id>")
def download_with_similars_pdf(report_id):
    if not session.get("logged_in"):
        return "Unauthorized", 401

    # current
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT result_text, embedding FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    if not row:
        cur.close(); conn.close()
        return "Not found", 404

    current_text = row["result_text"]
    emb = row["embedding"]
    if isinstance(emb, str):
        try:
            emb = json.loads(emb)
        except Exception:
            emb = None

    # similars
    similar_texts = []
    if emb:
        cur.execute("SELECT result_text, embedding FROM sreports WHERE id<>%s;", (report_id,))
        rows = cur.fetchall()
        for r in rows:
            vec = r["embedding"]
            if isinstance(vec, str):
                try:
                    vec = json.loads(vec)
                except Exception:
                    continue
            try:
                sim = cosine_similarity(emb, vec)
            except Exception:
                continue
            if sim >= 0.75:
                similar_texts.append(r["result_text"])

    cur.close(); conn.close()

    pdf = generate_pdf(current_text, similar_cases=similar_texts, title="Safety Report (+ Similar Cases)")
    return send_file(pdf, as_attachment=True, download_name="report_with_similars.pdf")

# ------------- run -------------
if __name__ == "__main__":
    app.run(debug=True)
