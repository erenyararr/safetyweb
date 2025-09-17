from flask import Flask, render_template_string, request, session, redirect, url_for, make_response
import fitz  # PyMuPDF
import openai
import datetime
import os
from functools import wraps

# --- API key ---
from config import API_KEY
openai.api_key = API_KEY

# --- Simple auth (hard-coded) ---
AUTH_USERNAME = "selectsafety"
AUTH_PASSWORD = "eren1234"

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
        messages=[{"role":"user","content":build_prompt(text, method, out_lang, feedback)}],
        max_tokens=1200
    )
    return resp.choices[0].message.content.strip()
    
