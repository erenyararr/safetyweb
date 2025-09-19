from flask import Flask, render_template_string, request, redirect, url_for, session, send_file
import fitz  # PyMuPDF
import openai
import datetime
import os
import io
import uuid
import json
import base64
import html
import re
import psycopg2
import psycopg2.extras
import numpy as np

# ====== OpenAI (openai==0.28.1) ======
from config import API_KEY
openai.api_key = API_KEY

# ====== App auth ======
USERNAME = "selectsafety"
PASSWORD = "eren1234"

# ====== DB ======
DB_URL = os.getenv("DATABASE_URL")

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
    """PDF içinden tüm metni çıkarır."""
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

def get_embedding(text: str):
    emb = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return emb["data"][0]["embedding"]

def cosine_similarity(v1, v2):
    a, b = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

STOPWORDS = {
  "the","a","an","of","to","and","or","in","on","at","for","with","by","from","as",
  "is","are","was","were","be","been","being","this","that","these","those",
  "it","its","into","over","under","while","during","due","than","then","so"
}

def tokenize_keywords(t: str):
    tokens = re.findall(r"[a-zA-Z]{3,}", t.lower())
    return [w for w in tokens if w not in STOPWORDS]

def extract_md_section(md: str, header: str) -> str:
    """
    Markdown içinde '### {header}' ile başlayan bölümü, bir sonraki başlığa kadar alır.
    Yoksa boş string döner.
    """
    lines = md.splitlines()
    start, buf = None, []
    for i, ln in enumerate(lines):
        if ln.strip().lower() == f"### {header}".lower():
            start = i + 1
            break
    if start is None:
        return ""
    for j in range(start, len(lines)):
        if lines[j].startswith("### "):
            break
        buf.append(lines[j])
    out = "\n".join(buf).strip()
    # Çok uzun ise kırp
    return out if len(out) <= 1200 else (out[:1200] + " …")
    
def why_similar(current: str, other: str):
    cur = set(tokenize_keywords(current))
    oth = set(tokenize_keywords(other))
    common = list(cur & oth)
    if not common:
        return "Operational context and contributing factors appear related."
    top = ", ".join(sorted(common, key=len, reverse=True)[:4])
    return f"Overlap on key terms: {top}. Sequences and contributing factors look comparable."

def build_prompt(report_text, method, out_lang, similar_items=None, feedback=None):
    lang_map = {
        "English": "Write the full analysis in clear, professional English.",
        "Français": "Rédige toute l’analyse en français professionnel et clair."
    }
    lang_line = lang_map.get(out_lang, lang_map["English"])

    similar_block = ""
    if similar_items:
        similar_block += "\n\n[Similar Past Cases Detected]\n"
        for i, s in enumerate(similar_items, 1):
            similar_block += (f"- Case {i} (score={s['score']:.2f}): "
                              f"{s['why']}\n"
                              f"  CASE INCIDENT SUMMARY: {s.get('incident_summary','(n/a)')}\n")

    prompt = f"""
You are an aviation safety analyst AI. Analyze the following safety report using the "{method}" method.

{lang_line}

Return a detailed markdown-formatted analysis with the following structure:

### Incident Summary
- Brief summary of the incident in 2-3 lines.

### Root Cause Analysis ({method})
- Explain the cause(s) of the incident using the selected method. Tie reasoning to any similar past cases if helpful.

### Short-term Solution (7 days)
- Actionable recommendations that can be implemented within a week.

### Long-term Solution (30 days)
- Preventative strategies and systemic improvements.

### Severity Level
- Categorize severity as: Minor / Moderate / Major / Critical

Here is the full new report text:
{report_text}
{similar_block}
"""
    if feedback:
        prompt += f"\n\n*** Reviewer Feedback to incorporate: ***\n{feedback}\n"
    return prompt

def analyze_with_gpt(text, method="Five Whys", out_lang="English", similar_items=None, feedback=None):
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": build_prompt(text, method, out_lang, similar_items, feedback)}],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()
# ====== PDF ======
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem
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

def generate_pdf(markdown_text: str, pdf_title: str = "Safety Report"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm, topMargin=20*mm, bottomMargin=16*mm,
        title=pdf_title
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"],
                                 alignment=TA_CENTER, fontName="Helvetica-Bold",
                                 fontSize=18, textColor=colors.HexColor("#22d3ee"),
                                 spaceAfter=10)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                        fontSize=14, textColor=colors.HexColor("#38bdf8"),
                        spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=10.5, leading=14.5, spaceAfter=6)

    elements = [Paragraph(pdf_title, title_style),
                HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#0ea5e9"), spaceAfter=8)]

    lines = [ln.rstrip() for ln in markdown_text.splitlines()]
    i, paragraph_buffer = 0, []

    def flush_para():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            elements.append(Paragraph(" ".join(paragraph_buffer).strip(), body))
            paragraph_buffer = []

    bullet_re = re.compile(r"^\s*[-•]\s+")
    num_re = re.compile(r"^\s*\d+\.\s+")

    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            flush_para(); elements.append(Spacer(1, 4)); i += 1; continue
        if ln.startswith("### "):
            flush_para()
            elements.append(Spacer(1, 2))
            elements.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#0ea5e9"), spaceAfter=4))
            elements.append(Paragraph(ln[4:].strip(), h2))
            i += 1; continue
        if bullet_re.match(ln):
            flush_para()
            items = []
            while i < len(lines) and bullet_re.match(lines[i]):
                txt = bullet_re.sub("", lines[i]).strip()
                items.append(ListItem(Paragraph(txt, body), leftIndent=6)); i += 1
            elements.append(ListFlowable(items, bulletType="bullet", leftPadding=12))
            elements.append(Spacer(1, 4)); continue
        if num_re.match(ln):
            flush_para()
            items = []
            while i < len(lines) and num_re.match(lines[i]):
                txt = num_re.sub("", lines[i]).strip()
                items.append(ListItem(Paragraph(txt, body), leftIndent=6)); i += 1
            elements.append(ListFlowable(items, bulletType="1", leftPadding=12))
            elements.append(Spacer(1, 4)); continue
        paragraph_buffer.append(ln); i += 1

    flush_para()
    doc.build(elements, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buf.seek(0)
    return buf.read()

# ====== UI ======
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
    .btn-primary { background:linear-gradient(90deg,#34d399,#22d3ee,#3b82f6); color:white; }
    .btn-primary:hover { filter:brightness(1.05); }
    a.btn-link { color:#67e8f9; text-decoration:underline; }
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
        <button class="px-3 py-2 rounded-lg bg-rose-500 text-white">Logout</button>
      </form>
      {% endif %}
    </div>

    {% if not session.get('logged_in') %}
      <div class="max-w-md mx-auto bg-white/10 p-8 rounded-3xl">
        <h2 class="text-2xl text-center mb-6">Login</h2>
        <form method="post" action="{{ url_for('login') }}" class="space-y-4">
          <input name="username" placeholder="Username" class="w-full p-3 rounded-lg bg-slate-800/70" required>
          <input name="password" type="password" placeholder="Password" class="w-full p-3 rounded-lg bg-slate-800/70" required>
          <button class="w-full py-3 rounded-xl btn-primary">Sign In</button>
        </form>
      </div>
    {% else %}
      <p class="text-slate-400 mb-6">Upload a PDF, we’ll find similar cases, and draft an analysis. You can send feedback and re-generate.</p>

      <form hx-post="{{ url_for('analyze') }}" hx-target="#reports" hx-swap="beforeend" enctype="multipart/form-data"
            class="bg-white/10 p-6 rounded-2xl mb-10">
        <label class="block text-lg font-semibold text-cyan-300 mb-2">Upload PDF</label>
        <input type="file" name="pdf" accept=".pdf" class="mb-4" required>
        <div class="flex gap-4 flex-wrap">
          <select name="method" class="flex-1 p-2 bg-slate-800/70 rounded-lg">
            <option>Five Whys</option>
            <option>Fishbone</option>
            <option>Bowtie</option>
          </select>
          <select name="lang" class="flex-1 p-2 bg-slate-800/70 rounded-lg">
            <option>English</option>
            <option>Français</option>
          </select>
          <button class="px-6 py-3 rounded-xl btn-primary">🚀 Run Analysis</button>
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

def render_similar_html(similar_items):
    """Benzer olayları (skor, why, Incident Summary) ve tam rapor linkini gösterir."""
    if not similar_items:
        return '<p class="text-slate-400">No similar cases found.</p>'
    out = []
    for s in similar_items:
        inc = s.get("incident_summary") or "(incident summary not available)"
        out.append(f"""
        <div class="p-3 rounded-lg bg-slate-900/50 border border-slate-700">
          <div class="font-semibold text-cyan-300">Similarity: {s['score']:.2f}</div>
          <div class="text-sm text-emerald-300 mb-1">Why similar: {html.escape(s['why'])}</div>
          <div class="text-sm text-slate-200 mb-2"><b>Incident Summary (from case):</b> {html.escape(inc)}</div>
          <div class="text-xs text-slate-400 mb-2"><b>Snippet:</b> {html.escape(s['snippet'])}</div>
          <a class="btn-link" href="{url_for('view_case', report_id=s['id'])}" target="_blank">Open full report →</a>
        </div>
        """)
    return "<div class='space-y-3'>" + "\n".join(out) + "</div>"

def render_similar_text(similar_items, include_full=False):
    """PDF metni: her case için skor, why, Incident Summary; istenirse full rapor da ekler."""
    if not similar_items:
        return "### Similar Cases\nNo similar cases."
    lines = ["### Similar Cases"]
    for i, s in enumerate(similar_items, 1):
        lines.append(f"## Case {i} — score={s['score']:.2f}")
        lines.append(f"Why similar: {s['why']}")
        if s.get("incident_summary"):
            lines.append("**Incident Summary (from case)**")
            lines.append(s["incident_summary"])
        if include_full and s.get("full_result"):
            lines.append("\n**Full Case Report**")
            lines.append(s["full_result"])
        lines.append("\n---\n")
    return "\n".join(lines)
@app.route("/analyze", methods=["POST"])
def analyze():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    pdf = request.files["pdf"]
    method = request.form.get("method", "Five Whys")
    lang = request.form.get("lang", "English")

    report_text = extract_text_from_pdf(pdf)
    emb = get_embedding(report_text)

    # DB: son 200 kaydı çek
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT id, report_text, result_text, embedding FROM sreports ORDER BY created_at DESC LIMIT 200;")
    rows = cur.fetchall()
    cur.close()
    conn.close()

    sims = []
    for r in rows:
        vec = r["embedding"]
        if vec is None:
            continue
        if isinstance(vec, str):
            try:
                vec = json.loads(vec)
            except Exception:
                continue
        try:
            score = cosine_similarity(emb, vec)
        except Exception:
            continue
        why = why_similar(report_text, r["report_text"] or r["result_text"] or "")
        full_result = r["result_text"] or ""
        inc_summary = extract_md_section(full_result, "Incident Summary")
        sims.append({
            "id": str(r["id"]),
            "score": score,
            "why": why,
            "snippet": (r["report_text"] or r["result_text"] or "")[:220].replace("\n"," ") + ("..." if (r["report_text"] or r["result_text"]) and len((r["report_text"] or r["result_text"]))>220 else ""),
            "incident_summary": inc_summary,
            "full_result": full_result
        })

    sims = sorted(sims, key=lambda x: x["score"], reverse=True)
    sims = [s for s in sims if s["score"] >= 0.70][:5]

    # GPT analizi
    result_text = analyze_with_gpt(report_text, method, lang, similar_items=sims)

    # DB'ye kaydet
    rid = str(uuid.uuid4())
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sreports (id, method, lang, report_text, result_text, embedding) VALUES (%s,%s,%s,%s,%s,%s);",
        (rid, method, lang, report_text, result_text, json.dumps(emb))
    )
    conn.commit()
    cur.close()
    conn.close()

    # PDF içerikleri
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf_only = f"# Safety Report — {method} — {lang}\n\n{result_text}"
    pdf_with_sim = (
        f"# Safety Report — {method} — {lang}\n\n{result_text}\n\n" +
        render_similar_text(sims, include_full=True)
    )
    b64_only = base64.b64encode(pdf_only.encode("utf-8")).decode("ascii")
    b64_with = base64.b64encode(pdf_with_sim.encode("utf-8")).decode("ascii")

    similar_html = render_similar_html(sims)

    block = f"""
    <div class="bg-slate-800/60 border border-slate-700 p-6 rounded-2xl">
      <div class="flex justify-between items-center mb-3">
        <h2 class="text-xl font-bold text-cyan-300">📄 Report ({method}, {lang})</h2>
        <span class="text-xs text-slate-400">{now}</span>
      </div>

      <div class="flex gap-3 mb-3 flex-wrap">
        <form action="{url_for('download_pdf')}" method="post" target="_blank">
          <input type="hidden" name="content_b64" value="{b64_only}">
          <input type="hidden" name="title" value="Safety Report — {method} — {lang}">
          <button class="px-3 py-2 rounded-lg bg-emerald-600 hover:brightness-110 text-white text-sm">⬇ Download PDF (report)</button>
        </form>
        <form action="{url_for('download_pdf')}" method="post" target="_blank">
          <input type="hidden" name="content_b64" value="{b64_with}">
          <input type="hidden" name="title" value="Safety Report + Similar — {method} — {lang}">
          <button class="px-3 py-2 rounded-lg bg-teal-600 hover:brightness-110 text-white text-sm">⬇ Download PDF (report + similar)</button>
        </form>
      </div>

      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700 mb-3">{html.escape(result_text)}</pre>

      <details class="bg-slate-900/50 border border-slate-700 rounded-lg p-3 mb-3">
        <summary class="cursor-pointer text-cyan-300 font-semibold">Show Similar Cases ({len(sims)})</summary>
        <div class="mt-3">{similar_html}</div>
      </details>

      <form hx-post="{url_for('feedback')}" hx-target="#reports" hx-swap="beforeend" class="space-y-2">
        <input type="hidden" name="method" value="{method}">
        <input type="hidden" name="lang" value="{lang}">
        <input type="hidden" name="orig" value="{html.escape(result_text)}">
        <input type="hidden" name="similar_b64" value="{base64.b64encode(json.dumps(sims).encode()).decode()}">
        <textarea name="feedback" rows="3" class="w-full p-2 rounded-lg bg-slate-900/60 border border-slate-700" placeholder="Give feedback..."></textarea>
        <button class="px-4 py-2 rounded-xl btn-primary">🔁 Update Report</button>
        <input type="hidden" name="text" value="{html.escape(report_text)}"/>
      </form>
    </div>
    """
    return block
@app.route("/case/<report_id>")
def view_case(report_id):
    """Benzer olaya tıklanınca tam raporu yeni sekmede gösterir + tek başına PDF indirir."""
    if not session.get("logged_in"):
        return "Unauthorized", 401
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT method, lang, result_text, created_at FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        return "Not found", 404
    method, lang, result_text, created_at = row["method"], row["lang"], row["result_text"], row["created_at"]
    title = f"Case {report_id[:8]} — {method} — {lang}"
    b64 = base64.b64encode(f"# {title}\n\n{result_text}".encode("utf-8")).decode("ascii")
    html_page = f"""
    <html><head>
      <meta charset="utf-8"/>
      <title>{html.escape(title)}</title>
      <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
      <style> body{{background:#0b1220;color:#d1e7ff;font-family:system-ui,Arial,sans-serif;padding:28px}}
      .card{{background:#0f172a;border:1px solid #1f2a44;border-radius:14px;padding:20px}}
      .btn{{display:inline-block;background:#10b981;color:#fff;padding:8px 12px;border-radius:10px;text-decoration:none}}
      pre{{white-space:pre-wrap;background:#0b1020;border:1px solid #253454;padding:12px;border-radius:10px}}
      </style>
    </head><body>
      <div class="card">
        <h2>{html.escape(title)}</h2>
        <div style="font-size:12px;opacity:.7;margin-bottom:8px;">Created: {created_at}</div>
        <form action="{url_for('download_pdf')}" method="post" target="_blank" style="margin-bottom:10px;">
          <input type="hidden" name="content_b64" value="{b64}">
          <input type="hidden" name="title" value="{html.escape(title)}">
          <button class="btn">⬇ Download PDF</button>
        </form>
        <pre>{html.escape(result_text)}</pre>
      </div>
    </body></html>
    """
    return html_page

@app.route("/feedback", methods=["POST"])
def feedback():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    method   = request.form.get("method","Five Whys")
    lang     = request.form.get("lang","English")
    text     = request.form.get("text","")
    fb       = request.form.get("feedback","")
    orig     = request.form.get("orig","")
    similar_b64 = request.form.get("similar_b64","")

    try:
        similar_items = json.loads(base64.b64decode(similar_b64.encode()).decode()) if similar_b64 else []
    except Exception:
        similar_items = []

    updated = analyze_with_gpt(text, method, lang, similar_items=similar_items, feedback=fb)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    pdf_updated = f"# Updated Safety Report — {method} — {lang}\n\n{updated}"
    pdf_full = (
        f"# Full Package — {method} — {lang}\n\n"
        f"### Original Report\n{orig}\n\n"
        f"### Updated Report (with feedback)\n{updated}\n\n" +
        render_similar_text(similar_items, include_full=True)
    )
    b64_updated = base64.b64encode(pdf_updated.encode("utf-8")).decode("ascii")
    b64_full = base64.b64encode(pdf_full.encode("utf-8")).decode("ascii")

    similar_html = render_similar_html(similar_items)

    block = f"""
    <div class="bg-slate-700/70 border border-slate-700 p-6 rounded-2xl">
      <div class="flex justify-between items-center mb-3">
        <h2 class="text-xl font-bold text-emerald-300">🤖 Updated Report ({method}, {lang})</h2>
        <span class="text-xs text-slate-300">{now}</span>
      </div>

      <div class="flex gap-3 mb-3 flex-wrap">
        <form action="{url_for('download_pdf')}" method="post" target="_blank">
          <input type="hidden" name="content_b64" value="{b64_updated}">
          <input type="hidden" name="title" value="Updated Report — {method} — {lang}">
          <button class="px-3 py-2 rounded-lg bg-emerald-600 hover:brightness-110 text-white text-sm">⬇ Download PDF (updated)</button>
        </form>
        <form action="{url_for('download_pdf')}" method="post" target="_blank">
          <input type="hidden" name="content_b64" value="{b64_full}">
          <input type="hidden" name="title" value="Full Package — {method} — {lang}">
          <button class="px-3 py-2 rounded-lg bg-teal-600 hover:brightness-110 text-white text-sm">⬇ Download PDF (original + updated + similar)</button>
        </form>
      </div>

      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700 mb-3">{html.escape(updated)}</pre>

      <details class="bg-slate-900/50 border border-slate-700 rounded-lg p-3">
        <summary class="cursor-pointer text-cyan-300 font-semibold">Show Similar Cases</summary>
        <div class="mt-3">{similar_html}</div>
      </details>
    </div>
    """
    return block

# Stateless PDF indirme — içerikten üretir (404 yok)
@app.route("/download", methods=["POST"])
def download_pdf():
    if not session.get("logged_in"):
        return "Unauthorized", 401
    b64 = request.form.get("content_b64","")
    title = request.form.get("title","Safety Report")
    try:
        content = base64.b64decode(b64.encode("ascii")).decode("utf-8")
    except Exception:
        return "Bad content", 400
    pdf_bytes = generate_pdf(content, pdf_title=title)
    return send_file(io.BytesIO(pdf_bytes), mimetype="application/pdf",
                     as_attachment=True, download_name=f"{title.replace(' ','_')}.pdf")

if __name__ == "__main__":
    app.run(debug=True)
