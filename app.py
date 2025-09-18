from flask import Flask, render_template_string, request, redirect, url_for, session, send_file
import fitz  # PyMuPDF
import openai
import datetime
import os
import io
import uuid
import re

# --- API key (openai==0.28.1 arayüzü) ---
from config import API_KEY
openai.api_key = API_KEY

# Kullanıcı girişi (gömülü)
USERNAME = "selectsafety"
PASSWORD = "eren1234"

# Üretilen PDF dosyalarını ID -> bytes olarak tutacağız (ephemeral)
PDF_STORE = {}  # {file_id: (filename, bytes)}

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
        messages=[{"role": "user", "content": build_prompt(text, method, out_lang, feedback)}],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()

# ---------- ŞIK PDF ÜRETİCİ (başlıklar, listeler, sarma, header/footer) ----------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors

def _header_footer(canvas, doc):
    canvas.saveState()
    # Header
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(colors.HexColor("#06b6d4"))
    canvas.drawString(doc.leftMargin, doc.height + doc.topMargin - 10, "✈ AI Safety Report Analyzer")
    # Footer
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(doc.leftMargin + doc.width, 12, f"Page {doc.page}")
    canvas.restoreState()

def generate_pdf(markdown_text: str, pdf_title: str = "Safety Report"):
    """
    Markdown benzeri metni şık bir PDF’e dönüştürür ve bytes döner.
    Başlıklar (###), bullet ve numaralı listeler, otomatik satır kaydırma, header/footer içerir.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=20 * mm,
        bottomMargin=16 * mm,
        title=pdf_title
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

    elements = []
    # Ana başlık
    elements.append(Paragraph(pdf_title, title_style))
    elements.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#0ea5e9"), spaceAfter=8))

    lines = [ln.rstrip() for ln in markdown_text.splitlines()]
    i = 0
    paragraph_buffer = []

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            p = Paragraph(" ".join(paragraph_buffer).strip(), body)
            elements.append(p)
            paragraph_buffer = []

    bullet_re = re.compile(r"^\s*[-•]\s+")
    numbered_re = re.compile(r"^\s*\d+\.\s+")

    while i < len(lines):
        ln = lines[i]

        if not ln.strip():  # boş satır
            flush_paragraph()
            elements.append(Spacer(1, 4))
            i += 1
            continue

        if ln.startswith("### "):
            flush_paragraph()
            elements.append(Spacer(1, 2))
            elements.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#0ea5e9"), spaceAfter=4))
            elements.append(Paragraph(ln[4:].strip(), h2))
            i += 1
            continue

        if bullet_re.match(ln):
            flush_paragraph()
            items = []
            while i < len(lines) and bullet_re.match(lines[i]):
                txt = bullet_re.sub("", lines[i]).strip()
                items.append(ListItem(Paragraph(txt, body), leftIndent=6))
                i += 1
            elements.append(ListFlowable(items, bulletType="bullet", leftPadding=12))
            elements.append(Spacer(1, 4))
            continue

        if numbered_re.match(ln):
            flush_paragraph()
            items = []
            while i < len(lines) and numbered_re.match(lines[i]):
                txt = numbered_re.sub("", lines[i]).strip()
                items.append(ListItem(Paragraph(txt, body), leftIndent=6))
                i += 1
            elements.append(ListFlowable(items, bulletType="1", leftPadding=12))
            elements.append(Spacer(1, 4))
            continue

        paragraph_buffer.append(ln)
        i += 1

    flush_paragraph()
    doc.build(elements, onFirstPage=_header_footer, onLaterPages=_header_footer)
    buf.seek(0)
    return buf.read()
# ======================
# Flask App
# ======================
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

# HTML (Tailwind + HTMX). Giriş yapılmadıysa login, yapıldıysa app görünür.
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
    .file-input-container { position: relative; display: inline-block; width: 100%; }
    .file-input-container input[type=file] { position: absolute; left: 0; top: 0; opacity: 0; width: 100%; height: 100%; cursor: pointer; }
    .file-input-label { display: block; background: #0d6efd; color: white; padding: 10px 16px; border-radius: 8px; font-weight: bold; text-align: center; cursor: pointer; }
  </style>
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
    u = request.form.get("username", "")
    p = request.form.get("password", "")
    if u == USERNAME and p == PASSWORD:
        session["logged_in"] = True
        return redirect(url_for("index"))
    # basitçe geri dönelim (prod'da flash kullanılır)
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
    method   = request.form.get("method","Five Whys")
    out_lang = request.form.get("lang","English")
    text     = extract_text_from_pdf(pdf_file)

    result = analyze_with_gpt(text, method, out_lang)

    # PDF üret (orijinal analiz)
    now_title = f"Safety Report — {method} — {out_lang}"
    pdf_bytes = generate_pdf(result, pdf_title=now_title)
    file_id = uuid.uuid4().hex
    PDF_STORE[file_id] = (f"report_{file_id}.pdf", pdf_bytes)
    download_url = url_for("download_pdf", file_id=file_id)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"""
    <div class="backdrop-blur-xl bg-slate-800/60 border border-cyan-400/20 shadow-lg p-6 rounded-2xl">
      <div class="flex justify-between items-center mb-3">
        <h2 class="text-xl font-bold text-cyan-300">📄 Report ({method}, {out_lang})</h2>
        <span class="text-xs text-slate-400">{now}</span>
      </div>

      <div class="flex gap-3 mb-3">
        <a href="{download_url}" class="px-3 py-2 rounded-lg bg-emerald-600/90 hover:bg-emerald-600 text-white text-sm">⬇ Download PDF</a>
      </div>

      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{result}</pre>

      <!-- Feedback form -->
      <form hx-post="{url_for('feedback')}" hx-target="#reports" hx-swap="beforeend" class="mt-4 space-y-2">
        <input type="hidden" name="method" value="{method}"/>
        <input type="hidden" name="lang" value="{out_lang}"/>
        <textarea name="feedback" rows="3" class="w-full border border-cyan-400/30 rounded-lg p-2 bg-slate-900/60 text-slate-200" placeholder="Give feedback..."></textarea>
        <button class="bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600 text-white px-4 py-2 rounded-xl hover:scale-105 transition">
          🔁 Update Report
        </button>
        <input type="hidden" name="text" value="{text}"/>
      </form>
    </div>
    """
    return block

@app.route("/feedback", methods=["POST"])
def feedback():
    if not session.get("logged_in"):
        return "Unauthorized", 401

    method   = request.form.get("method","Five Whys")
    out_lang = request.form.get("lang","English")
    text     = request.form.get("text","")
    fb       = request.form.get("feedback","")

    updated = analyze_with_gpt(text, method, out_lang, feedback=fb)

    # PDF üret (feedbackli)
    now_title = f"Updated Safety Report — {method} — {out_lang}"
    pdf_bytes = generate_pdf(updated, pdf_title=now_title)
    file_id = uuid.uuid4().hex
    PDF_STORE[file_id] = (f"updated_report_{file_id}.pdf", pdf_bytes)
    download_url = url_for("download_pdf", file_id=file_id)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    block = f"""
    <div class="backdrop-blur-xl bg-slate-700/70 border border-emerald-400/30 shadow-lg p-6 rounded-2xl">
      <div class="flex justify-between items-center mb-3">
        <h2 class="text-xl font-bold text-emerald-300">🤖 Updated Report ({method}, {out_lang})</h2>
        <span class="text-xs text-slate-400">{now}</span>
      </div>

      <div class="flex gap-3 mb-3">
        <a href="{download_url}" class="px-3 py-2 rounded-lg bg-emerald-600/90 hover:bg-emerald-600 text-white text-sm">⬇ Download PDF</a>
      </div>

      <pre class="whitespace-pre-wrap text-sm bg-slate-900/60 p-3 rounded border border-slate-700">{updated}</pre>
    </div>
    """
    return block
@app.route("/download/<file_id>")
def download_pdf(file_id):
    if not session.get("logged_in"):
        return "Unauthorized", 401
    if file_id not in PDF_STORE:
        return "Not found", 404
    filename, data = PDF_STORE[file_id]
    return send_file(
        io.BytesIO(data),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=filename
    )

if __name__ == "__main__":
    # Local’de test
    app.run(debug=True)
