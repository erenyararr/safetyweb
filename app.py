from flask import Flask, render_template_string, request, send_file  # send_file eklendi
import fitz  # PyMuPDF
import openai
import datetime
import os

# --- PDF export için ek importlar ---
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# --- API key ---
from config import API_KEY
openai.api_key = API_KEY
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
    resp = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":build_prompt(text, method, out_lang, feedback)}],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()

# ============== PDF Export Helper (YENİ) ==============
def generate_pdf(content: str):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    text_obj = c.beginText(40, height - 40)
    text_obj.setFont("Helvetica", 10)

    for line in (content or "").splitlines():
        # sayfa sonu kontrolü
        if text_obj.getY() <= 40:
            c.drawText(text_obj)
            c.showPage()
            text_obj = c.beginText(40, height - 40)
            text_obj.setFont("Helvetica", 10)
        text_obj.textLine(line)

    c.drawText(text_obj)
    c.save()
    buffer.seek(0)
    return buffer
# ======================
# Flask App
# ======================
app = Flask(__name__)

# Ana HTML template (Tailwind + HTMX, tek sayfa, custom file input)
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
    .file-input-container {
      position: relative;
      display: inline-block;
      width: 100%;
    }
    .file-input-container input[type=file] {
      position: absolute;
      left: 0;
      top: 0;
      opacity: 0;
      width: 100%;
      height: 100%;
      cursor: pointer;
    }
    .file-input-label {
      display: block;
      background: #0d6efd;
      color: white;
      padding: 10px 16px;
      border-radius: 8px;
      font-weight: bold;
      text-align: center;
      cursor: pointer;
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
@app.route("/")
def index():
    return render_template_string(TEMPLATE)

@app.route("/analyze", methods=["POST"])
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

      <!-- Download original PDF (YENİ) -->
      <form action="/download_original" method="post" target="_blank" class="mt-2">
        <textarea name="content" class="hidden">{result}</textarea>
        <button class="px-4 py-2 bg-indigo-500 text-white rounded-lg">⬇️ Download Original PDF</button>
      </form>

      <!-- Feedback form -->
      <form hx-post="/feedback" hx-target="#reports" hx-swap="beforeend" 
            class="mt-4 space-y-2">
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

      <!-- Download updated PDF (YENİ) -->
      <form action="/download_feedback" method="post" target="_blank" class="mt-2">
        <textarea name="content" class="hidden">{updated}</textarea>
        <button class="px-4 py-2 bg-green-600 text-white rounded-lg">⬇️ Download Updated PDF</button>
      </form>
    </div>
    """
    return block

# ====== Yeni indirme endpoint'leri ======
@app.route("/download_original", methods=["POST"])
def download_original():
    content = request.form.get("content", "")
    pdf = generate_pdf(content)
    return send_file(pdf, as_attachment=True, download_name="original_report.pdf")

@app.route("/download_feedback", methods=["POST"])
def download_feedback():
    content = request.form.get("content", "")
    pdf = generate_pdf(content)
    return send_file(pdf, as_attachment=True, download_name="feedback_report.pdf")

if __name__ == "__main__":
    app.run(debug=True)
