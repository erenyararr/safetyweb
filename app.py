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

# --- API key (openai==0.28.1 arayüzü) ---
from config import API_KEY
openai.api_key = API_KEY

# Kullanıcı girişi
USERNAME = "selectsafety"
PASSWORD = "eren1234"

# DB bağlantısı
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
        extra += "\n\n📌 The AI has detected similar past safety reports (context snippets):\n"
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

def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1, dtype=float), np.array(v2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

def explain_similarity_brief(current_text: str, past_text: str) -> str:
    """
    1–2 doğal İngilizce cümle. 'These two events are similar because ...' ile başlar.
    """
    prompt = f"""
You are an aviation safety analyst.

Write 1–2 plain English sentences that explain why the following two incidents are similar.
Start the answer EXACTLY with: "These two events are similar because "
Mention the strongest shared hazards/causes, triggers, context, or outcomes.
Keep it under 45 words. No bullets, no lists, no headings, no quotes, no markdown.

Current incident:
{current_text}

Past incident:
{past_text}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

def analyze_with_gpt(text, method="Five Whys", out_lang="English", feedback=None, similar_cases=None):
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": build_prompt(text, method, out_lang, feedback, similar_cases)}],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()

# ==================== PDF EXPORT ====================
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

def generate_pdf(current_report, similar_blocks=None, title="Safety Report"):
    """
    similar_blocks: list of dicts like
      [{"score": 0.83, "why": "...", "result_text": "full past analysis"}, ...]
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Current Report</b>", styles["Heading2"]))
    story.append(Paragraph(current_report.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 10))

    if similar_blocks:
        story.append(Paragraph("<b>Similar Cases</b>", styles["Heading2"]))
        for i, blk in enumerate(similar_blocks, 1):
            sc = f"{blk.get('score', 0):.2f}"
            story.append(Paragraph(f"<b>Case {i} (Similarity: {sc})</b>", styles["Heading3"]))
            if blk.get("why"):
                story.append(Paragraph(blk["why"], styles["BodyText"]))
            if blk.get("result_text"):
                story.append(Spacer(1, 4))
                story.append(Paragraph(blk["result_text"].replace("\n", "<br/>"), styles["BodyText"]))
            story.append(Spacer(1, 10))

    doc.build(story)
    buf.seek(0)
    return buf
# ==================== FLASK ====================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

# HTML (Show Similar Cases + iki ayrı PDF butonu)
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
    button:hover { opacity: .95; }
    details > summary { cursor: pointer; }
    .btn { padding:.6rem 1rem; border-radius:.7rem; color:white; font-weight:600; }
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
          <button class="btn" style="background:#ef4444">Logout</button>
        </form>
      {% endif %}
    </div>

    {% if not session.get('logged_in') %}
      <div class="max-w-md mx-auto bg-white/10 p-8 rounded-3xl">
        <h2 class="text-2xl text-center mb-6">Login</h2>
        <form method="post" action="{{ url_for('login') }}" class="space-y-4">
          <input name="username" placeholder="Username" class="w-full p-3 rounded-lg bg-slate-800/70" required>
          <input name="password" type="password" placeholder="Password" class="w-full p-3 rounded-lg bg-slate-800/70" required>
          <button class="w-full py-3 btn" style="background:#06b6d4">Sign In</button>
        </form>
      </div>
    {% else %}
      <p class="text-slate-400 mb-8">Analyze reports with AI, find similar past cases, export polished PDFs.</p>

      <form hx-post="{{ url_for('analyze') }}" hx-target="#reports" hx-swap="beforeend" enctype="multipart/form-data"
            class="bg-white/10 p-8 rounded-3xl mb-10">
        <label class="block text-lg font-semibold text-cyan-300 mb-2">Upload PDF</label>
        <input type="file" name="pdf" accept=".pdf" required class="mb-4">
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
          <button class="btn" style="background:#22c55e">🚀 Run Analysis</button>
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
def _load_all_reports():
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT id, report_text, result_text, embedding FROM sreports ORDER BY created_at DESC;")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def _parse_embedding(val):
    if val is None:
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return None
    if isinstance(val, (dict,)):
        # some drivers may store JSONB as python objects already
        return val if isinstance(val, list) else None
    return None

def _top5_similars(current_emb, current_text):
    rows = _load_all_reports()
    scored = []
    for r in rows:
        vec = _parse_embedding(r["embedding"])
        if not vec:
            continue
        sc = cosine_similarity(current_emb, vec)
        scored.append((sc, r["id"], r["report_text"], r["result_text"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    # ilk 5
    top = scored[:5]
    # neden benzer açıklamaları üret
    similar_blocks = []
    for sc, rid, rpt_txt, res_txt in top:
        why = explain_similarity_brief(current_text, rpt_txt)
        similar_blocks.append({
            "id": rid,
            "score": float(sc),
            "why": why,
            "report_text": rpt_txt,
            "result_text": res_txt
        })
    return similar_blocks

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

    # Prompt için benzerlerden kısa konteks (top-5)
    similar_blocks = _top5_similars(emb, text)
    similar_for_prompt = [blk["result_text"] for blk in similar_blocks]

    # AI analiz
    result = analyze_with_gpt(text, method, lang, similar_cases=similar_for_prompt)

    # DB’ye kaydet
    rid = str(uuid.uuid4())
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("""INSERT INTO sreports (id, method, lang, report_text, result_text, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s);""",
                (rid, method, lang, text, result, json.dumps(emb)))
    conn.commit()
    cur.close()
    conn.close()

    # HTML blok
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"""
    <div class="bg-slate-800/60 p-6 rounded-2xl">
      <div class="flex justify-between mb-3">
        <h2 class="text-xl font-bold text-cyan-300">📄 Report ({method}, {lang})</h2>
        <span class="text-xs">{now}</span>
      </div>

      <div class="flex flex-wrap gap-3 mb-3">
        <a href="{{{{ url_for('download_pdf', report_id='{rid}') }}}}" class="btn" style="background:#10b981">⬇ PDF (Current)</a>
        <a href="{{{{ url_for('download_pdf', report_id='{rid}', include_sim='1') }}}}" class="btn" style="background:#14b8a6">⬇ PDF + Similar Cases</a>
        <button class="btn" style="background:#3b82f6"
                hx-get="{{{{ url_for('similar_cases', report_id='{rid}') }}}}"
                hx-target="#sims-{rid}" hx-swap="innerHTML">
          🔎 Show Similar Cases (Top 5)
        </button>
      </div>

      <pre class="whitespace-pre-wrap text-sm">{result}</pre>

      <form hx-post="{{{{ url_for('feedback') }}}}" hx-target="#reports" hx-swap="beforeend" class="mt-3">
        <input type="hidden" name="report_id" value="{rid}">
        <textarea name="feedback" class="w-full p-2 bg-slate-800/70 rounded-lg" placeholder="Give feedback..."></textarea>
        <button class="btn mt-2" style="background:#06b6d4">🔁 Update Report</button>
      </form>

      <div id="sims-{rid}" class="mt-4"></div>
    </div>
    """
    return block
@app.route("/similar/<report_id>")
def similar_cases(report_id):
    # Bu raporun embedding'ini ve metnini çek
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, embedding FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return "<div class='text-rose-400'>Report not found.</div>"

    emb = _parse_embedding(row["embedding"])
    if emb is None:
        emb = get_embedding(row["report_text"])

    blocks = _top5_similars(emb, row["report_text"])

    if not blocks:
        return "<div class='text-slate-300'>No similar cases found.</div>"

    html = ["<div class='bg-slate-900/50 p-4 rounded-xl border border-slate-700'>",
            "<h3 class='text-emerald-300 font-bold mb-3'>Top 5 Similar Cases</h3>"]
    for i, b in enumerate(blocks, 1):
        html.append(f"""
        <div class="mb-4 p-3 rounded-lg bg-slate-800/60">
          <div class="flex items-center justify-between">
            <div class="font-semibold">Case {i} — Similarity: {b['score']:.2f}</div>
          </div>
          <div class="text-sm text-slate-200 mt-1"><i>{b['why']}</i></div>
          <details class="mt-2">
            <summary class="text-sky-300">View full past report</summary>
            <pre class="whitespace-pre-wrap text-sm mt-2">{b['result_text']}</pre>
          </details>
        </div>
        """)
    html.append("</div>")
    return "\n".join(html)

@app.route("/feedback", methods=["POST"])
def feedback():
    rid = request.form.get("report_id")
    fb = request.form.get("feedback","").strip()
    if not fb:
        return "<div class='text-amber-400 mt-2'>No feedback provided.</div>"

    # Orijinal raporu al
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, method, lang FROM sreports WHERE id=%s;", (rid,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return "<div class='text-rose-400 mt-2'>Report not found.</div>"

    updated = analyze_with_gpt(row["report_text"], row["method"], row["lang"], feedback=fb)

    # DB'yi güncelle (son sürüm result_text olsun)
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("UPDATE sreports SET result_text=%s WHERE id=%s;", (updated, rid))
    conn.commit()
    cur.close()
    conn.close()

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""
    <div class="bg-slate-700/70 p-4 rounded-lg mt-3">
      <div class="flex justify-between mb-2">
        <h3 class="text-emerald-300 font-bold">🤖 Updated Report</h3>
        <span class="text-xs text-slate-400">{now}</span>
      </div>
      <pre class="whitespace-pre-wrap text-sm">{updated}</pre>
    </div>
    """

@app.route("/download/<report_id>")
def download_pdf(report_id):
    include_sim = request.args.get("include_sim") == "1"

    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, result_text, embedding FROM sreports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return "Not found", 404

    current = row["result_text"]  # ekranda gördüğün rapor (güncel)
    similar_blocks = None
    if include_sim:
        emb = _parse_embedding(row["embedding"]) or get_embedding(row["report_text"])
        blocks = _top5_similars(emb, row["report_text"])
        # PDF için sadece score/why/result_text gerekli
        similar_blocks = [{"score": b["score"], "why": b["why"], "result_text": b["result_text"]} for b in blocks]

    pdf = generate_pdf(current_report=current, similar_blocks=similar_blocks, title="Safety Report")
    return send_file(pdf, as_attachment=True, download_name="report.pdf")

if __name__ == "__main__":
    app.run(debug=True)
