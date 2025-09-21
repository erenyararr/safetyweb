# scripts/ingest_cadors.py
"""
CADORS CSV'yi (son 24 ay vb.) okuyup 'sreports' tablosuna aktarır ve
CADORS numarasını 'cadors_index' tablosunda sreports.id ile ilişkilendirir.

Kullanım (Railway'de ya da lokal):
    python scripts/ingest_cadors.py --csv data/cadors_last24m.csv           # sadece insert (embedding yok)
    python scripts/ingest_cadors.py --csv data/cadors_last24m.csv --embed   # insert + OpenAI embedding
    python scripts/ingest_cadors.py --csv data/cadors_last24m.csv --embed --max-rows 1000

Gereken ENV:
    DATABASE_URL       -> PostgreSQL bağlantısı
    OPENAI_API_KEY     -> (sadece --embed verirsen) OpenAI 0.28.x için API key

Notlar:
- 'sreports.method' = 'Imported (CADORS)' olarak eklenir (UI'da ayırt etmesi kolay).
- Aynı CADORS numarası birden fazla kez eklenmesin diye 'cadors_index' tablosu kullanılır.
"""

import os, sys, csv, uuid, json, argparse, datetime as dt, re
import psycopg2
import psycopg2.extras

# OpenAI 0.28.x uyumu (app.py ile aynı stil)
import openai

# ---- ENV & Globals ----
DB_URL = os.getenv("DATABASE_URL")
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or os.getenv("API_KEY")

# ---- SQL ----
SQL_INIT = """
CREATE TABLE IF NOT EXISTS sreports (
    id UUID PRIMARY KEY,
    method TEXT,
    lang TEXT,
    report_text TEXT,
    result_text TEXT,
    embedding JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS cadors_index (
    cadors_no TEXT PRIMARY KEY,
    sreports_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""

SQL_SELECT_INDEX = "SELECT sreports_id FROM cadors_index WHERE cadors_no = %s;"
SQL_INSERT_REPORT = """
INSERT INTO sreports (id, method, lang, report_text, result_text, embedding, created_at)
VALUES (%s, %s, %s, %s, %s, %s, NOW());
"""
SQL_INSERT_INDEX  = "INSERT INTO cadors_index (cadors_no, sreports_id) VALUES (%s, %s);"

# ---- Helpers ----
def connect():
    if not DB_URL:
        print("ERROR: DATABASE_URL yok.", file=sys.stderr)
        sys.exit(1)
    return psycopg2.connect(DB_URL, sslmode="require")

def init_tables():
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(SQL_INIT)
        conn.commit()

def norm_header(h: str) -> str:
    return re.sub(r"\s+", " ", (h or "").strip()).lower()

def get(row, *keys, default="") -> str:
    """
    CSV kolon isimleri değişebildiği için esnek al.
    örn: ('Cadors Number','CADORS Number','CADORS No')
    """
    lowered = {norm_header(k): v for k, v in row.items()}
    for k in keys:
        v = lowered.get(norm_header(k))
        if v not in (None, ""):
            return str(v).strip()
    return default

def build_report_text(row: dict) -> str:
    # Alanlar (esnek başlık eşleşmesi)
    cad = get(row, "Cadors Number", "CADORS Number", "CADORS No", "CADORS #")
    odate = get(row, "Occurrence Date", "Event Date", "Date")
    otime = get(row, "Occurrence Time", "Time")
    otype = get(row, "Occurrence Type", "Type")
    aerodrome = get(row, "Aerodrome Name", "Aerodrome")
    location  = get(row, "Occurrence Location", "Location")
    province  = get(row, "Province")
    country   = get(row, "Country")
    ac_reg    = get(row, "Registration", "Aircraft Registration")
    ac_make   = get(row, "Make", "Aircraft Make")
    ac_model  = get(row, "Model", "Aircraft Model")
    phase     = get(row, "Phase of Flight", "Phase")
    operator  = get(row, "Operator")
    summary   = get(row, "All Narrative (Delimited by Date)", "Narrative", "Summary")
    # Markdown benzeri sade metin (app.pdf renderer bunu güzel basıyor)
    parts = [
        f"### CADORS {cad}",
        f"- Occurrence Date/Time: {odate or '-'} {otime or ''}".strip(),
        f"- Type: {otype or '-'}",
        f"- Location: {location or '-'} / {aerodrome or '-'} / {province or '-'} / {country or '-'}",
        f"- Aircraft: {ac_reg or '-'} / {ac_make or '-'} {ac_model or ''}".strip(),
        f"- Phase: {phase or '-'}",
        f"- Operator: {operator or '-'}",
        "",
        "### Incident Summary",
        summary or "-",
    ]
    return "\n".join(parts).strip(), cad

def get_embedding(text: str):
    # OpenAI 0.28.x
    resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return resp["data"][0]["embedding"]

def upsert_row(row: dict, do_embed: bool) -> tuple[bool, bool]:
    """
    Returns: (inserted, duplicate)
    """
    report_text, cad_no = build_report_text(row)
    if not cad_no:
        # boş/bozuk satır
        return (False, False)

    with connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(SQL_SELECT_INDEX, (cad_no,))
            hit = cur.fetchone()
            if hit:
                # zaten var
                return (False, True)

            rid = uuid.uuid4()
            emb = None
            if do_embed:
                if not OPENAI_KEY:
                    raise RuntimeError("OPENAI_API_KEY yok; --embed için gerekli.")
                openai.api_key = OPENAI_KEY
                try:
                    emb = get_embedding(report_text)
                except Exception as e:
                    # embedding hatası olursa yine de kaydı tut (embedding NULL kalsın)
                    print(f"[WARN] embedding fail for {cad_no}: {e}", file=sys.stderr)
                    emb = None

            cur.execute(
                SQL_INSERT_REPORT,
                (
                    str(rid),
                    "Imported (CADORS)",
                    "English",
                    report_text,
                    "",          # result_text boş, istersen daha sonra app içinde üretirsin
                    json.dumps(emb) if emb is not None else None,
                ),
            )
            cur.execute(SQL_INSERT_INDEX, (cad_no, str(rid)))
        conn.commit()

    return (True, False)

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="Import CADORS CSV -> sreports (+cadors_index)")
    p.add_argument("--csv", required=True, help="CSV path")
    p.add_argument("--embed", action="store_true", help="OpenAI embedding üret")
    p.add_argument("--max-rows", type=int, default=None, help="En fazla şu kadar satır işle")
    return p.parse_args()

def main():
    args = parse_args()

    if args.embed:
        if not OPENAI_KEY:
            print("ERROR: --embed verildi ama OPENAI_API_KEY yok.", file=sys.stderr)
            sys.exit(1)
        openai.api_key = OPENAI_KEY
    else:
        print("[INFO] running WITHOUT embeddings (use --embed to enable)")

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV bulunamadı: {args.csv}", file=sys.stderr)
        sys.exit(1)

    init_tables()

    total = inserted = skipped = dup = 0
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            try:
                ok, is_dup = upsert_row(row, do_embed=args.embed)
                if ok:
                    inserted += 1
                elif is_dup:
                    dup += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                print(f"[WARN] insert fail on row#{total}: {e}", file=sys.stderr)

            if args.max_rows and total >= args.max_rows:
                break

    print(f"DONE. total={total} inserted={inserted} skipped={skipped} dup={dup}")

if __name__ == "__main__":
    main()
