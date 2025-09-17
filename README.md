# SafetyWeb – AI Safety Report Analyzer

A simple Flask app to analyze aviation safety reports from PDF using OpenAI. Upload a PDF, pick an analysis method (Five Whys, Fishbone, Bowtie), and get a structured, markdown-style analysis. Includes a feedback loop to refine the output.

## Quickstart

1) Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Configure environment

- Copy `.env.example` to `.env` and fill in your key:

```bash
OPENAI_API_KEY=your_openai_key_here
```

4) Run the app

```bash
python app.py
```

App runs at `http://127.0.0.1:5000/`.

## Notes

- Secrets are loaded via `python-dotenv`. Do not commit your real `.env`.
- PDF text is extracted with PyMuPDF (`fitz`).
- The web UI uses Tailwind and HTMX via CDNs.

## Deploy

- Set environment variable `OPENAI_API_KEY` on your host.
- Use a production WSGI server (e.g., `gunicorn`) behind a reverse proxy for production.

