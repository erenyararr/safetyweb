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

# --- API key ---
from config import API_KEY
openai.api_key = API_KEY

# ====== Auth (embedded) ======
USERNAME = "selectsafety"
PASSWORD = "eren1234"

# ====== DB ======
DB_URL = os.getenv("DATABASE_URL")

# Benzerlik ayarları
SIM_THRESHOLD = 0.75
SIM_TOP_K = 5

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

# ====== Helpers ======
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
        extra += "\n\n📌 The AI has detected similar past safety reports and will consider them while analyzing:\n"
        for case in similar_cases[:SIM_TOP_K]:
            # raporun ilk 200 karakterini göster
            snippet = case.replace("\n", " ")[:200]
            extra += f"- {snippet}...\n"

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

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
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

# ====== Pretty PDF (başlıklar & listeler) ======
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

def render_markdown_like(text):
    """Basit markdown benzeri: ### başlık, - bullet, 1. numaralı listeler"""
    styles = getSampleStyleSheet()
    h2 = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#38bdf8"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=14.5,
        spaceAfter=6,
    )

    lines = [ln.rstrip() for ln in text.splitlines()]
    i = 0
    elements = []
    paragraph_buffer = []

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            p = Paragraph(" ".join(paragraph_buffer).strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"), body)
            elements.append(p)
            paragraph_buffer = []

    bullet_re = re.compile(r"^\s*[-•]\s+")
    numbered_re = re.compile(r"^\s*\d+\.\s+")

    while i < len(lines):
        ln = lines[i]

        if not ln.strip():
            flush_paragraph()
            elements.append(Spacer(1, 4))
            i += 1
            continue

        if ln.startswith("### "):
            flush_paragraph()
            elements.append(Spacer(1, 2))
            elements.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#0ea5e9"), spaceAfter=4))
            elements.append(Paragraph(ln[4:].strip().replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"), h2))
            i += 1
            continue

        if bullet_re.match(ln):
            flush_paragraph()
            items = []
            while i < len(lines) and bullet_re.match(lines[i]):
                txt = bullet_re.sub("", lines[i]).strip()
                items.append(ListItem(Paragraph(txt.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"), body), leftIndent=6))
                i += 1
            elements.append(ListFlowable(items, bulletType="bullet", leftPadding=12))
            elements.append(Spacer(1, 4))
            continue

        if numbered_re.match(ln):
            flush_paragraph()
            items = []
            while i < len(lines) and numbered_re.match(lines[i]):
                txt = numbered_re.sub("", lines[i]).strip()
                items.append(ListItem(Paragraph(txt.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;"), body), leftIndent=6))
                i += 1
            elements.append(ListFlowable(items, bulletType="1", leftPadding=12))
            elements.append(Spacer(1, 4))
            continue

        paragraph_buffer.append(ln)
        i += 1

    flush_paragraph()
    return elements

def generate_pdf(current_report, similar_cases, title="Safety Report"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18*mm,
        rightMargin=18*mm,
        topMargin=20*mm,
        bottomMargin=16*mm,
        title=title
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=colors.HexColor("#22d3ee"),
        spaceAfter=10,
        spaceBefore=0,
    )

    story = []
    story.append(Paragraph(title, title_style))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#0ea5e9"), spaceAfter=8))

    # Current
    story += render_markdown_like("### Current Report\n" + current_report)

    # Similar
    if similar_cases:
        story += render_markdown_like("### Similar Cases")
        for i, case in enumerate(similar_cases, 1):
            story += render_markdown_like(f"### Case {i}\n{case}")

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buf.seek(0)
    return buf

# ====== Flask ======
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
  <style>
    button:active { transform: scale(0.97); }
    button:hover { filter: brightness(1.05); }
    .file-input-container { position: relative; width: 100%; }
    .file-input { position:absolute; inset:0; opacity:0; cursor:pointer; }
    .file-label { display:flex; align-items:center; justify-content:center; gap:.5rem;
                  background:#0ea5e9; color:white; padding:.6rem 1rem; border-radius:.75rem; font-weight:700; }
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
          <button class="bg-rose-500 px-3 py-2 rounded-lg text-white">Logout</button>
        </form>
      {% endif %}
    </div>

    {% if not session.get('logged_in') %}
      <div class="max-w-md mx-auto backdrop-blur-lg bg-white/10 border border-white/20 shadow-2xl p-8 rounded-3xl">
        <h2 class="text-2xl text-center mb-6">Login</h2>
        <form method="post" action="{{ url_for('login') }}" class="space-y-4">
          <input name="username" placeholder="Username" class="w-full p-3 rounded-lg bg-slate-800/70 border border-slate-700" required>
          <input name="password" type="password" placeholder="Password" class="w-full p-3 rounded-lg bg-slate-800/70 border border-slate-700" required>
          <button class="w-full py-3 rounded-xl font-bold text-white bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500">
            Sign In
          </button>
        </form>
      </div>
    {% else %}
      <p class="text-slate-400 mb-8">Analyze reports with AI, find similar past cases, and export sleek PDFs.</p>

      <form hx-post="{{ url_for('analyze') }}" hx-target="#reports" hx-swap="beforeend" enctype="multipart/form-data"
            class="backdrop-blur-lg bg-white/10 border border-white/20 shadow-2xl p-6 rounded-2xl mb-10 space-y-5">
        <div>
          <label class="block text-lg font-semibold text-cyan-300 mb-2">Upload PDF</label>
          <div class="file-input-container">
            <span class="file-label">📄 Choose File</span>
            <input class="file-input" type="file" name="pdf" accept=".pdf" required>
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
          <button class="px-6 py-3 rounded-xl font-bold text-white bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500">
            🚀 Run Analysis
          </button>
        </div>
      </form>

      <div id="reports" class="space-y-8"></div>
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

def find_similar_cases(query_embedding):
    """DB'deki tüm raporlar içinde cosine similarity ile benzerleri döndürür."""
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT id, report_text, result_text, embedding FROM sreports ORDER BY created_at DESC;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    scored = []
    for r in rows:
        emb_obj = r["embedding"]
        if emb_obj is None:
            continue
        # JSONB psycopg2 ile genelde Python list olarak gelir; string ise json.loads
        vec = emb_obj if isinstance(emb_obj, list) else json.loads(emb_obj)
        try:
            sim = cosine_similarity(query_embedding, vec)
        except Exception:
            continue
        if sim >= SIM_THRESHOLD:
            scored.append({
                "id": str(r["id"]),
                "sim": sim,
                "text": r["result_text"] or r["report_text"] or ""
            })

    # Skora göre sırala ve top-K al
    scored.sort(key=lambda x: x["sim"], reverse=True)
    return scored[:SIM_TOP_K]

@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    pdf_file = request.files["pdf"]
    method   = request.form.get("method","Five Whys")
    lang     = request.form.get("lang","English")
    text     = extract_text_from_pdf(pdf_file)

    # Embedding
    emb = get_embedding(text)

    # Similarları bul
    sims = find_similar_cases(emb)
    similar_texts = [s["text"] for s in sims]

    # AI analiz
    result = analyze_with_gpt(text, method, lang, similar_cases=similar_texts)

    # Kaydet
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

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    # Kart: PDF indir, Similar Cases butonu, Feedback formu (her biri ayrı HTMX çağrısı)
    block = f"""
    <div class="backdrop-blur-xl bg-slate-800/60 border border-cyan-400/20 shadow-lg p-6 rounded-2xl">
      <div class="flex flex-wrap items-center justify-between gap-3 mb-4">
        <h2 class="text-xl font-bold text-cyan-300">📄 Report ({method}, {lang})</h2>
        <span class="text-xs text-slate-400">{now}</span>
      </div>

      <div class="flex flex-wrap gap-3 mb-4">
        <a href="{{{{ url_for('download_pdf', report_id='{rid}', include_sim='0') }}}}" class="px-3 py-2 rounded-lg bg-emerald-600/90 hover:bg-emerald-600 text-white text-sm">⬇ Download PDF (Current)</a>
        <a href="{{{{ url_for('download_pdf', report_id='{rid}', include_sim='1') }}}}" class="px-3 py-2 rounded-lg bg-teal-600/90 hover:bg-teal-600 text-white text-sm">⬇ Download PDF (+ Similar)</a>
        <button hx-post="{{{{ url_for('show_similar') }}}}" hx-vals='{{"report_id":"{rid}"}}' hx-target="#sim-{rid}" class="px-3 py-2 rounded-lg bg-sky-600/90 hover:bg-sky-600 text-white text-sm">
          🔎 Find Similar Cases
        </button>
      </div>

      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{result}</pre>

      <div id="sim-{rid}" class="mt-4"></div>

      <form hx-post="{{{{ url_for('feedback') }}}}" hx-target="#reports" hx-swap="beforeend" class="mt-5 space-y-2">
        <input type="hidden" name="report_id" value="{rid}">
        <textarea name="feedback" rows="3" class="w-full border border-cyan-400/30 rounded-lg p-2 bg-slate-900/60 text-slate-200" placeholder="Give feedback..."></textarea>
        <button class="bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600 text-white px-4 py-2 rounded-xl">
          🔁 Update Report
        </button>
      </form>
    </div>
    """
    return block

@app.route("/similar", methods=["POST"])
def show_similar():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    rid = request.form.get("report_id", "")
    # report embedding’ini çek
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT embedding FROM sreports WHERE id=%s;", (rid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row or row["embedding"] is None:
        return "<div class='text-slate-400'>No similar cases found.</div>"

    base_emb = row["embedding"] if isinstance(row["embedding"], list) else json.loads(row["embedding"])
    sims = find_similar_cases(base_emb)

    if not sims:
        return "<div class='text-slate-400'>No similar cases found.</div>"

    # Liste olarak döndür
    items = []
    for s in sims:
        score = f"{s['sim']:.2f}"
        snippet = (s["text"] or "").strip().replace("<","&lt;").replace(">","&gt;")
        short = snippet[:600] + ("..." if len(snippet) > 600 else "")
        items.append(f"""
          <div class="bg-slate-900/50 border border-slate-700 rounded-lg p-3">
            <div class="text-xs text-slate-400 mb-1">Similarity: <span class="text-emerald-400 font-mono">{score}</span></div>
            <pre class="whitespace-pre-wrap text-sm">{short}</pre>
          </div>
        """)

    return f"""
      <div class="mt-3 space-y-3">
        <h3 class="text-cyan-300 font-semibold">Similar Cases</h3>
        {''.join(items)}
      </div>
    """

@app.route("/feedback", methods=["POST"])
def feedback():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    rid = request.form.get("report_id")
    fb  = request.form.get("feedback","").strip()

    # Orijinal raporu getir
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, method, lang, embedding FROM sreports WHERE id=%s;", (rid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return "<div class='text-rose-400'>Report not found.</div>"

    # Similarları yeniden bul (aynı eşik/logic)
    base_emb = row["embedding"] if isinstance(row["embedding"], list) else json.loads(row["embedding"])
    sims = find_similar_cases(base_emb)
    similar_texts = [s["text"] for s in sims]

    # Feedback ile güncel analiz (yeni blok olarak appended)
    updated = analyze_with_gpt(row["report_text"], row["method"], row["lang"], feedback=fb, similar_cases=similar_texts)

    # DB’ye güncelle
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("UPDATE sreports SET result_text=%s WHERE id=%s;", (updated, rid))
    conn.commit()
    cur.close()
    conn.close()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    <div class="backdrop-blur-xl bg-slate-700/70 border border-emerald-400/30 shadow-lg p-6 rounded-2xl">
      <div class="flex items-center justify-between mb-3">
        <h3 class="text-emerald-300 font-bold">🤖 Updated Report</h3>
        <span class="text-xs text-slate-400">{now}</span>
      </div>
      <div class="flex flex-wrap gap-3 mb-3">
        <a href="{{{{ url_for('download_pdf', report_id='{rid}', include_sim='0') }}}}" class="px-3 py-2 rounded-lg bg-emerald-600/90 hover:bg-emerald-600 text-white text-sm">⬇ Download PDF (Current)</a>
        <a href="{{{{ url_for('download_pdf', report_id='{rid}', include_sim='1') }}}}" class="px-3 py-2 rounded-lg bg-teal-600/90 hover:bg-teal-600 text-white text-sm">⬇ Download PDF (+ Similar)</a>
      </div>
      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{updated}</pre>
    </div>
    """

@app.route("/download/<report_id>")
def download_pdf(report_id):
    if not session.get("logged_in"):
        return "Unauthorized", 401

    include_sim = request.args.get("include_sim", "0") == "1"

    # current text
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT result_text, embedding FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return "Not found", 404

    current = row["result_text"] or ""
    similar_list = []

    if include_sim:
        base_emb = row["embedding"] if isinstance(row["embedding"], list) else json.loads(row["embedding"])
        sims = find_similar_cases(base_emb)
        similar_list = [s["text"] for s in sims]

    pdf = generate_pdf(current, similar_list, title="Safety Report")
    return send_file(pdf, mimetype="application/pdf", as_attachment=True, download_name="report.pdf")

if __name__ == "__main__":
    app.run(debug=True)
