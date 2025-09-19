from flask import Flask, render_template_string, request, redirect, url_for, session, send_file
import fitz  # PyMuPDF
import openai
import datetime
import os
import uuid
import psycopg2
import psycopg2.extras
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# --- API key ---
from config import API_KEY
openai.api_key = API_KEY

# Kullanıcı girişi
USERNAME = "selectsafety"
PASSWORD = "eren1234"

# Postgres bağlantısı
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
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()
# ================== HELPERS ===================
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
# ================= PDF EXPORT ===================
def generate_pdf(current_report, similar_cases, title="Safety Report"):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Current Report:</b>", styles["Heading2"]))
    story.append(Paragraph(current_report.replace("\n", "<br/>"), styles["BodyText"]))
    story.append(Spacer(1, 12))

    if similar_cases:
        story.append(Paragraph("<b>Similar Cases:</b>", styles["Heading2"]))
        for idx, case in enumerate(similar_cases, 1):
            story.append(Paragraph(f"<b>Case {idx}:</b>", styles["Heading3"]))
            story.append(Paragraph(case.replace("\n", "<br/>"), styles["BodyText"]))
            story.append(Spacer(1, 12))

    doc.build(story)
    buf.seek(0)
    return buf
# ================= FLASK ===================
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
</head>
<body class="bg-gradient-to-br from-slate-900 via-sky-900 to-slate-800 text-slate-200 min-h-screen">
  <div class="max-w-6xl mx-auto py-10 px-6">
    <h1 class="text-4xl font-extrabold text-cyan-400 mb-6">✈️ AI Safety Report Analyzer</h1>

    {% if not session.get('logged_in') %}
      <form method="post" action="{{ url_for('login') }}" class="bg-white/10 p-8 rounded-3xl max-w-md mx-auto">
        <input name="username" placeholder="Username" class="w-full p-3 rounded-lg bg-slate-800/70 mb-3">
        <input name="password" type="password" placeholder="Password" class="w-full p-3 rounded-lg bg-slate-800/70 mb-3">
        <button class="w-full py-3 bg-gradient-to-r from-emerald-400 via-cyan-500 to-blue-500 rounded-xl">Sign In</button>
      </form>
    {% else %}
      <form hx-post="{{ url_for('analyze') }}" hx-target="#reports" hx-swap="beforeend" enctype="multipart/form-data"
            class="bg-white/10 p-8 rounded-3xl mb-8">
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
      <div id="reports"></div>
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

    # Benzer olayları bul (çok basit: son 3 raporu benzer kabul ediyoruz)
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT result_text FROM reports ORDER BY created_at DESC LIMIT 3;")
    rows = cur.fetchall()
    similar = [r["result_text"] for r in rows]
    cur.close()
    conn.close()

    result = analyze_with_gpt(text, method, lang, similar_cases=similar)

    # Kaydet
    rid = str(uuid.uuid4())
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("INSERT INTO reports (id, method, lang, report_text, result_text) VALUES (%s,%s,%s,%s,%s);",
                (rid, method, lang, text, result))
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
      <a href="{{{{ url_for('download_pdf', report_id='{rid}') }}}}" 
         class="bg-emerald-500 px-3 py-2 rounded-lg text-white">⬇ Download PDF</a>
      <pre class="whitespace-pre-wrap text-sm">{result}</pre>
      <form hx-post="{{{{ url_for('feedback') }}}}" hx-target="#reports" hx-swap="beforeend" class="mt-3">
        <input type="hidden" name="report_id" value="{rid}">
        <textarea name="feedback" class="w-full p-2 bg-slate-800/70 rounded-lg" placeholder="Give feedback..."></textarea>
        <button class="bg-cyan-500 px-3 py-2 rounded-lg text-white mt-2">🔁 Update Report</button>
      </form>
    </div>
    """
    return block

@app.route("/feedback", methods=["POST"])
def feedback():
    rid = request.form.get("report_id")
    fb = request.form.get("feedback","")

    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT report_text, method, lang FROM reports WHERE id=%s;", (rid,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    updated = analyze_with_gpt(row["report_text"], row["method"], row["lang"], feedback=fb)

    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    cur.execute("UPDATE reports SET result_text=%s WHERE id=%s;", (updated, rid))
    conn.commit()
    cur.close()
    conn.close()

    return f"<div class='bg-slate-700 p-4 rounded-lg mt-2'><b>Updated Report:</b><pre>{updated}</pre></div>"

@app.route("/download/<report_id>")
def download_pdf(report_id):
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT result_text FROM reports WHERE id=%s;", (report_id,))
    row = cur.fetchone()
    cur.execute("SELECT result_text FROM reports ORDER BY created_at DESC LIMIT 3;")
    sims = [r["result_text"] for r in cur.fetchall()]
    cur.close()
    conn.close()

    pdf = generate_pdf(row["result_text"], sims)
    return send_file(pdf, as_attachment=True, download_name="report.pdf")

if __name__ == "__main__":
    app.run(debug=True)
