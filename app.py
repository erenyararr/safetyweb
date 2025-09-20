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

# ---------- OpenAI (0.28.x) ----------
from config import API_KEY
openai.api_key = API_KEY

# ---------- DB ----------
DB_URL = os.getenv("DATABASE_URL")

def init_db():
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()

    # Raporlar
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

    # Kullanıcılar (manuel dolduracaksın)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS susers (
        id UUID PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,          -- düz metin (kolay kurulum için)
        can_see_similar BOOLEAN DEFAULT TRUE,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)

    conn.commit()
    cur.close()
    conn.close()

init_db()

# ---------- Helpers ----------
STOP = set("""
the and for with that this from into over under across about above below between during after before while would could
have has had were was are is been being also only more most less least very much many such those these then than
shall will may might can cannot not nor but yet though although because since due therefore however hence
""".split())

def extract_text_from_pdf(pdf_file) -> str:
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def get_embedding(text: str):
    emb = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return emb["data"][0]["embedding"]

def cosine_similarity(v1, v2) -> float:
    a, b = np.array(v1, dtype=float), np.array(v2, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)

def top_keywords(text: str, k=10):
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    words = [w for w in words if w not in STOP]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:k]]

def incident_summary_from_markdown(md: str) -> str:
    m = re.search(r"(?im)^###\s*Incident Summary\s*\n(.+?)(?:\n###|\Z)", md, re.S)
    if not m:
        return "\n".join(md.strip().splitlines()[:4])
    return m.group(1).strip()

def build_why_similar(curr_text: str, past_text: str, overlap_terms, sim_score: float) -> str:
    """
    2–3 cümlelik, uyarı içeren kısa açıklama üretir (model çağrısı yapmadan).
    """
    overlap_part = ""
    if overlap_terms:
        overlap_part = f"Key term overlap detected: {', '.join(overlap_terms[:6])}. "
    else:
        overlap_part = "High-level pattern alignment (phase, contributing factors) appears present. "

    warn = "Note: This is a heuristic similarity; operational context (aircraft, airport, crew, weather) may differ."
    nuance = "Use these parallels to inspire mitigation ideas, not as one-to-one prescriptions."
    return f"{overlap_part}Similarity score ≈ {sim_score:.2f}. {warn} {nuance}"

def build_prompt(text, method, out_lang, feedback=None, similar_cases=None):
    lang_map = {
        "English": "Write the full analysis in clear, professional English.",
        "Français": "Rédige toute l’analyse en français professionnel et clair."
    }
    lang_line = lang_map.get(out_lang, lang_map["English"])

    extra = ""
    if similar_cases:
        extra += "\n\n📌 The AI has detected similar past safety reports:\n"
        for c in similar_cases:
            extra += (
                f"- (Similarity {c['sim']:.2f}) {c['snippet']}\n"
                f"    Why similar: {c['why']}\n"
            )

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
# ---------- PDF ----------
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
    title_style = ParagraphStyle("Title2", parent=styles["Heading1"], alignment=TA_CENTER,
                                 fontName="Helvetica-Bold", fontSize=18,
                                 textColor=colors.HexColor("#22d3ee"), spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                        fontSize=14, textColor=colors.HexColor("#38bdf8"),
                        spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=10.5, leading=14.5, spaceAfter=6)
    return title_style, h2, body

def _render_simple_markdown(elements, markdown_text, h2, body):
    lines = [ln.rstrip() for ln in markdown_text.splitlines()]
    i, para_buf = 0, []

    def flush_p():
        nonlocal para_buf
        if para_buf:
            elements.append(Paragraph(" ".join(para_buf).strip(), body))
            para_buf = []

    bullet_re = re.compile(r"^\s*[-•]\s+")
    numbered_re = re.compile(r"^\s*\d+\.\s+")
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            flush_p(); elements.append(Spacer(1,4)); i += 1; continue
        if ln.startswith("### "):
            flush_p(); elements.append(Spacer(1,2))
            elements.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#0ea5e9"), spaceAfter=4))
            elements.append(Paragraph(ln[4:].strip(), h2)); i += 1; continue
        if bullet_re.match(ln):
            flush_p(); items=[]
            while i < len(lines) and bullet_re.match(lines[i]):
                txt = bullet_re.sub("", lines[i]).strip()
                items.append(ListItem(Paragraph(txt, body), leftIndent=6)); i += 1
            elements.append(ListFlowable(items, bulletType="bullet", leftPadding=12)); elements.append(Spacer(1,4)); continue
        if numbered_re.match(ln):
            flush_p(); items=[]
            while i < len(lines) and numbered_re.match(lines[i]):
                txt = numbered_re.sub("", lines[i]).strip()
                items.append(ListItem(Paragraph(txt, body), leftIndent=6)); i += 1
            elements.append(ListFlowable(items, bulletType="1", leftPadding=12)); elements.append(Spacer(1,4)); continue
        para_buf.append(ln); i += 1
    flush_p()

def generate_pdf_report(markdown_text: str, title="Safety Report"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=20*mm, bottomMargin=16*mm, title=title)
    title_style, h2, body = _mk_styles()
    elements = [Paragraph(title, title_style),
                HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#0ea5e9"), spaceAfter=8)]
    _render_simple_markdown(elements, markdown_text, h2, body)
    doc.build(elements, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buf.seek(0)
    return buf

def generate_pdf_full(current_markdown, similar_list, title="Safety Report (with similar)"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=20*mm, bottomMargin=16*mm, title=title)
    title_style, h2, body = _mk_styles()
    els = [Paragraph(title, title_style),
           HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#0ea5e9"), spaceAfter=8)]

    els.append(Paragraph("Current Report", h2))
    els.append(Paragraph(current_markdown.replace("\n", "<br/>"), body))
    els.append(Spacer(1,10))

    if similar_list:
        els.append(Paragraph("Similar Cases", h2))
        for i, c in enumerate(similar_list, 1):
            els.append(Paragraph(f"Case {i} — Similarity: {c['sim']:.2f}", body))
            els.append(Paragraph(f"Why similar: {c['why']}", body))
            els.append(Paragraph(f"Snippet:", body))
            els.append(Paragraph(c["snippet"].replace("\n", "<br/>"), body))
            els.append(Spacer(1,6))
            if c.get("full_markdown"):
                els.append(Paragraph("<b>Full Case Report</b>", body))
                els.append(Paragraph(c["full_markdown"].replace("\n", "<br/>"), body))
                els.append(Spacer(1,10))

    doc.build(els, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buf.seek(0)
    return buf
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
    button[disabled] { opacity: .6; cursor: not-allowed; }
  </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-slate-900 via-sky-900 to-slate-800 text-slate-200">
  <div class="max-w-6xl mx-auto py-10 px-6">

    <div class="flex justify-between items-center mb-4">
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
      <div class="max-w-md mx-auto bg-white/10 p-8 rounded-3xl">
        <h2 class="text-2xl text-center mb-6">Login</h2>
        <form method="post" action="{{ url_for('login') }}" class="space-y-4">
          <input name="username" placeholder="Username" class="w-full p-3 rounded-lg bg-slate-800/70" required>
          <input name="password" type="password" placeholder="Password" class="w-full p-3 rounded-lg bg-slate-800/70" required>
          <button class="w-full py-3 bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 rounded-xl">Sign In</button>
        </form>
      </div>
    {% else %}
      <p class="text-slate-400 mb-10">Upload a PDF, we'll find similar cases, and draft an analysis. You can send feedback and re-generate.</p>

      <!-- Upload form (English-only custom file input) -->
      <form hx-post="{{ url_for('analyze') }}" hx-target="#reports" hx-swap="beforeend" enctype="multipart/form-data"
            class="bg-white/10 p-8 rounded-3xl mb-12 space-y-6">

        <div>
          <label class="block text-lg font-semibold text-cyan-300 mb-2">Upload PDF</label>

          <div class="relative inline-flex items-center">
            <!-- hidden real input -->
            <input id="pdfInput" name="pdf" type="file" accept=".pdf"
                   class="absolute inset-0 w-full h-full opacity-0 cursor-pointer" required
                   onchange="document.getElementById('fileName').textContent=this.files?.[0]?.name||'No file chosen'">

            <!-- visible English button -->
            <button type="button" class="px-4 py-2 rounded-lg text-white"
                    style="background:linear-gradient(90deg,#34d399,#22d3ee,#3b82f6);">
              📄 Choose File
            </button>

            <!-- filename -->
            <span id="fileName" class="ml-3 text-sm text-slate-300">No file chosen</span>
          </div>
        </div>

        <div class="flex flex-wrap gap-4">
          <select name="method" class="flex-1 p-2 bg-slate-800/70 rounded-lg">
            <option>Five Whys</option>
            <option>Fishbone</option>
            <option>Bowtie</option>
          </select>
          <select name="lang" class="flex-1 p-2 bg-slate-800/70 rounded-lg">
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
    username = request.form.get("username","").strip()
    password = request.form.get("password","").strip()

    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT id, password, can_see_similar, is_active FROM susers WHERE username=%s;", (username,))
    row = cur.fetchone()
    cur.close(); conn.close()

    if row and row["is_active"] and row["password"] == password:
        session["logged_in"] = True
        session["user_id"] = str(row["id"])
        session["username"] = username
        session["can_see_similar"] = bool(row["can_see_similar"])
    # Başarısızsa sessizce ana sayfaya dön (UI sade kalsın)
    return redirect(url_for("index"))

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("logged_in", None)
    session.pop("user_id", None)
    session.pop("username", None)
    session.pop("can_see_similar", None)
    return redirect(url_for("index"))

def _fetch_all_reports():
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT id, report_text, result_text, embedding FROM sreports ORDER BY created_at DESC LIMIT 1000;")
    rows = cur.fetchall()
    cur.close(); conn.close()
    return rows
@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    pdf_file = request.files["pdf"]
    method   = request.form.get("method","Five Whys")
    lang     = request.form.get("lang","English")
    text     = extract_text_from_pdf(pdf_file)

    # 1) Embedding & benzer adaylar
    q_emb = get_embedding(text)
    curr_terms = top_keywords(text)
    candidates = []
    for r in _fetch_all_reports():
        if r["embedding"]:
            try:
                vec = json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"]
                sim = cosine_similarity(q_emb, vec)
                # eşiği düşük tutup sonradan top5 seçeceğiz
                if sim >= 0.60:
                    past_text = r["report_text"] or ""
                    overlap = list(set(curr_terms) & set(top_keywords(past_text)))
                    why = build_why_similar(text, past_text, overlap, sim)
                    summ = incident_summary_from_markdown(r["result_text"] or r["report_text"] or "")
                    candidates.append({
                        "id": str(r["id"]),
                        "sim": sim,
                        "snippet": (summ or past_text[:220]).strip(),
                        "why": why,
                        "full_markdown": r["result_text"] or ""
                    })
            except Exception:
                pass

    # 2) Gerçekten en benzer 5
    similar_cases = sorted(candidates, key=lambda x: -x["sim"])[:5]

    # 3) Analizi AI ile yaz (benzerler dahil)
    result = analyze_with_gpt(text, method, lang, similar_cases=similar_cases)

    # 4) DB'ye kaydet
    rid = str(uuid.uuid4())
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("INSERT INTO sreports (id, method, lang, report_text, result_text, embedding) VALUES (%s,%s,%s,%s,%s,%s);",
                (rid, method, lang, text, result, json.dumps(q_emb)))
    conn.commit(); cur.close(); conn.close()

    # 5) UI bloğu
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sim_target = f"sim-{rid}"
    can_see = session.get("can_see_similar", True)
    sim_btn = ""
    if can_see:
        sim_btn = f"""
        <button class="px-3 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm"
                hx-get="{url_for('similar_cases', report_id=rid)}"
                hx-target="#{sim_target}" hx-swap="innerHTML">🔎 Show Similar Cases</button>
        """

    block = f"""
    <div class="bg-slate-800/60 p-6 rounded-2xl border border-white/10">
      <div class="flex justify-between items-center mb-3">
        <h2 class="text-xl font-bold text-cyan-300">📄 Report ({method}, {lang})</h2>
        <span class="text-xs text-slate-400">{now}</span>
      </div>

      <div class="flex flex-wrap gap-3 mb-3">
        <a class="px-3 py-2 rounded-lg bg-emerald-600/90 hover:bg-emerald-600 text-white text-sm"
           href="{url_for('download_report', report_id=rid)}">⬇ Download PDF (report)</a>
        <a class="px-3 py-2 rounded-lg bg-sky-600/90 hover:bg-sky-600 text-white text-sm"
           href="{url_for('download_full', report_id=rid)}">⬇ Download PDF (report + similar)</a>
        {sim_btn}
      </div>

      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{result}</pre>

      <div id="{sim_target}" class="mt-4"></div>

      <form hx-post="{url_for('feedback')}" hx-target="#reports" hx-swap="beforeend" class="mt-4 space-y-2">
        <input type="hidden" name="report_id" value="{rid}"/>
        <input type="hidden" name="method" value="{method}"/>
        <input type="hidden" name="lang" value="{lang}"/>
        <input type="hidden" name="text" value="{text.replace('"','&quot;')}"/>
        <textarea name="feedback" rows="3" class="w-full border border-cyan-400/30 rounded-lg p-2 bg-slate-900/60 text-slate-200" placeholder="Give feedback..."></textarea>
        <button class="bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600 text-white px-4 py-2 rounded-xl">
          🔁 Update Report
        </button>
      </form>
    </div>
    """
    return block

@app.route("/similar/<report_id>")
def similar_cases(report_id):
    # Mevcut raporu al
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, result_text, embedding FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        return "<div class='text-rose-400'>Not found.</div>"

    text = row["report_text"] or ""
    q_emb = json.loads(row["embedding"]) if isinstance(row["embedding"], str) else row["embedding"]
    curr_terms = top_keywords(text)

    items = []
    for r in _fetch_all_reports():
        if str(r["id"]) == report_id:
            continue
        if r["embedding"]:
            try:
                vec = json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"]
                sim = cosine_similarity(q_emb, vec)
                if sim >= 0.60:
                    past_txt = r["report_text"] or ""
                    overlap = list(set(curr_terms) & set(top_keywords(past_txt)))
                    why = build_why_similar(text, past_txt, overlap, sim)
                    summ = incident_summary_from_markdown(r["result_text"] or r["report_text"] or "")
                    items.append((sim, str(r["id"]), summ, why))
            except Exception:
                pass

    items.sort(key=lambda x: -x[0])
    items = items[:5]  # UI'da da en iyi 5

    if not items:
        return "<div class='text-slate-300'>No close matches found.</div>"

    html = ["<div class='bg-slate-900/40 p-3 rounded-lg border border-white/10'>",
            "<div class='font-semibold text-cyan-300 mb-2'>Similar Cases (Top 5)</div>"]
    for sim, cid, summ, why in items:
        html.append(f"""
        <div class="mb-3 p-3 rounded-lg bg-slate-800/50 border border-slate-700">
          <div class="text-emerald-300 font-semibold">Similarity: {sim:.2f}</div>
          <div class="text-emerald-200 text-sm mb-1"><b>Why similar:</b> {why}</div>
          <div class="text-slate-300 text-sm"><b>Snippet:</b> {summ[:500]}</div>
          <div class="mt-2">
            <a href="{url_for('case_fullpage', case_id=cid)}" target="_blank"
               class="text-sky-300 underline">Open full case in new tab</a>
            <button class="ml-3 px-2 py-1 text-xs rounded bg-slate-700 hover:bg-slate-600"
                    hx-get="{url_for('case_preview', case_id=cid)}"
                    hx-target="#prev-{cid}" hx-swap="innerHTML">Preview here</button>
          </div>
          <div id="prev-{cid}" class="mt-2"></div>
        </div>
        """)
    html.append("</div>")
    return "\n".join(html)

@app.route("/case/preview/<case_id>")
def case_preview(case_id):
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT result_text FROM sreports WHERE id=%s;", (case_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        return "<div class='text-rose-400'>Not found.</div>"
    return f"<pre class='whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700'>{row['result_text']}</pre>"

@app.route("/case/<case_id>")
def case_fullpage(case_id):
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT result_text FROM sreports WHERE id=%s;", (case_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        return "Not found", 404
    return f"<html><body style='background:#0f172a;color:#e2e8f0;font-family:ui-sans-serif;padding:20px'><h2>Case {case_id}</h2><pre style='white-space:pre-wrap;background:#0b1220;padding:12px;border-radius:8px'>{row['result_text']}</pre></body></html>"
@app.route("/feedback", methods=["POST"])
def feedback():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    rid     = request.form.get("report_id")
    method  = request.form.get("method", "Five Whys")
    lang    = request.form.get("lang", "English")
    text    = request.form.get("text","")
    fb      = request.form.get("feedback","")

    updated = analyze_with_gpt(text, method, lang, feedback=fb)

    # PDF (updated) in-memory
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf_buf = generate_pdf_report(updated, title=f"Updated Safety Report — {method} — {lang}")
    file_id = uuid.uuid4().hex
    _PDF_STORE[file_id] = (f"updated_report_{file_id}.pdf", pdf_buf.getvalue())

    block = f"""
    <div class="bg-slate-700/70 p-4 rounded-2xl border border-white/10">
      <div class="flex justify-between mb-2">
        <h3 class="text-emerald-300 font-bold">🤖 Updated Report</h3>
        <span class="text-xs text-slate-400">{now}</span>
      </div>
      <div class="mb-2">
        <a class="px-3 py-2 rounded-lg bg-emerald-600/90 hover:bg-emerald-600 text-white text-sm"
           href="{url_for('download_memory_pdf', file_id=file_id)}">⬇ Download PDF (updated)</a>
      </div>
      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{updated}</pre>
    </div>
    """
    return block

# In-memory store for updated downloads
_PDF_STORE = {}  # {file_id: (filename, bytes)}

@app.route("/download/report/<report_id>")
def download_report(report_id):
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT result_text, method, lang FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        return "Not found", 404

    pdf = generate_pdf_report(row["result_text"], title=f"Safety Report — {row['method']} — {row['lang']}")
    return send_file(pdf, as_attachment=True, download_name="report.pdf")

@app.route("/download/full/<report_id>")
def download_full(report_id):
    # report + similar cases
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, result_text, embedding FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        return "Not found", 404

    current_md = row["result_text"] or ""
    text = row["report_text"] or ""
    q_emb = json.loads(row["embedding"]) if isinstance(row["embedding"], str) else row["embedding"]
    curr_terms = top_keywords(text)

    # recompute top-5 with explanations
    sims = []
    for r in _fetch_all_reports():
        if str(r["id"]) == report_id:
            continue
        if r["embedding"]:
            try:
                vec = json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"]
                sim = cosine_similarity(q_emb, vec)
                if sim >= 0.60:
                    past_txt = r["report_text"] or ""
                    overlap = list(set(curr_terms) & set(top_keywords(past_txt)))
                    why = build_why_similar(text, past_txt, overlap, sim)
                    summ = incident_summary_from_markdown(r["result_text"] or r["report_text"] or "")
                    sims.append({
                        "id": str(r["id"]),
                        "sim": sim,
                        "snippet": summ or past_txt[:220],
                        "why": why,
                        "full_markdown": r["result_text"] or ""
                    })
            except Exception:
                pass
    sims.sort(key=lambda x: -x[["sim"] if isinstance(x, dict) else 0])
    sims.sort(key=lambda x: -x["sim"])  # ensure correct
    sims = sims[:5]

    pdf = generate_pdf_full(current_md, sims, title="Safety Report (with Similar Cases)")
    return send_file(pdf, as_attachment=True, download_name="report_with_similar.pdf")

@app.route("/download/memory/<file_id>")
def download_memory_pdf(file_id):
    if file_id not in _PDF_STORE:
        return "Not found", 404
    fname, data = _PDF_STORE[file_id]
    return send_file(io.BytesIO(data), mimetype="application/pdf", as_attachment=True, download_name=fname)

# Short aliases
@app.route("/d/<report_id>")
def d_short(report_id):
    return download_report(report_id)

@app.route("/df/<report_id>")
def df_short(report_id):
    return download_full(report_id)

if __name__ == "__main__":
    app.run(debug=True)
