#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADORS CSV (son 24 ay vb.) -> Postgres `sreports`
- delimiter otomatik (`;` veya `,`)
- "Occurrence Information" üst satırı atlar
- kolon adları case-insensitive
- tekrar eklemeyi `cadors_index` ile engeller
- embedding opsiyonel (--embed)

Kullanım (Railway worker):
  python scripts/ingest_cadors.py --csv data/cadors_last24m.csv
  # isteğe bağlı:
  # python scripts/ingest_cadors.py --csv data/cadors_last24m.csv --embed --max-rows 2000
"""

import os, sys, csv, json, uuid, argparse, datetime as dt
import psycopg2, psycopg2.extras
import openai

# ---------- Config / Env ----------
API_KEY = os.getenv("OPENAI_API_KEY")
try:
    if not API_KEY:
        from config import API_KEY as _FALLBACK
        API_KEY = _FALLBACK
except Exception:
    pass
if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

openai.api_key = API_KEY

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("Missing DATABASE_URL")

# ---------- DB ----------
def get_conn():
    return psycopg2.connect(DB_URL, sslmode="require")

def init_tables():
    ddl_sreports = """
    CREATE TABLE IF NOT EXISTS sreports (
        id UUID PRIMARY KEY,
        method TEXT,
        lang TEXT,
        report_text TEXT,
        result_text TEXT,
        embedding JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );"""
    ddl_cadors_index = """
    CREATE TABLE IF NOT EXISTS cadors_index (
        cadors_no TEXT PRIMARY KEY,
        sreports_id UUID NOT NULL,
        source TEXT DEFAULT 'CADORS',
        created_at TIMESTAMP DEFAULT NOW()
    );"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(ddl_sreports); cur.execute(ddl_cadors_index)
        conn.commit()

def cadors_exists(cadors_no: str) -> bool:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM cadors_index WHERE cadors_no=%s;", (cadors_no,))
        return cur.fetchone() is not None

def link_cadors(cadors_no: str, report_id: str):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO cadors_index (cadors_no, sreports_id)
                       VALUES (%s,%s) ON CONFLICT (cadors_no) DO NOTHING;""",
                    (cadors_no, report_id))
        conn.commit()

def insert_sreport(method: str, lang: str, report_text: str, embedding):
    rid = str(uuid.uuid4())
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO sreports
                       (id, method, lang, report_text, result_text, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s);""",
                    (rid, method, lang, report_text, "", json.dumps(embedding) if embedding is not None else None))
        conn.commit()
    return rid

# ---------- Embedding ----------
def get_embedding(text: str):
    try:
        r = openai.Embedding.create(model="text-embedding-3-small", input=text)
        return r["data"][0]["embedding"]
    except Exception as e:
        print(f"[WARN] embedding failed: {e}", file=sys.stderr)
        return None

# ---------- CSV helpers ----------
def smart_rows(path):
    """CSV'yi oku, delimiteri kokla, doğru header satırını bul ve dict satırlar üret."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
            delim = dialect.delimiter
        except Exception:
            # Türkçe Windows genellikle ';'
            delim = ";" if sample.count(";") >= sample.count(",") else ","

        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)

    # Header satırını bul (Cadors + Number içeren)
    header_idx = 0
    for i, r in enumerate(rows[:5]):
        joined = " ".join(r).lower()
        if "cadors" in joined and "number" in joined:
            header_idx = i
            break

    header = [h.strip() for h in rows[header_idx]]
    data   = rows[header_idx+1:]

    for r in data:
        if not any(x.strip() for x in r):
            continue
        d = { (header[i] if i < len(header) else f"col{i}"): (r[i].strip() if i < len(r) else "")
              for i in range(max(len(header), len(r))) }
        yield d

def safe(d: dict, key: str) -> str:
    """Case-insensitive sözlük erişimi."""
    lk = key.lower()
    for k, v in d.items():
        if (k or "").strip().lower() == lk:
            return (v or "").strip()
    return ""

def build_report_text(row: dict) -> str:
    cad_no = safe(row, "Cadors Number")
    odate  = safe(row, "Occurrence Date")
    otype  = safe(row, "Occurrence Type")
    aero   = safe(row, "Aerodrome Name")
    loc    = safe(row, "Occurrence Location")
    prov   = safe(row, "Province")
    narr   = safe(row, "All Narrative (Delimited by Date)") or safe(row, "Narrative")

    place = aero or loc or "Unknown"
    title = f"CADORS {cad_no} — {otype} on {odate} at {place}" + (f" ({prov})" if prov else "")

    bits = []
    if otype: bits.append(f"Occurrence Type: {otype}")
    if odate: bits.append(f"Date: {odate}")
    if aero:  bits.append(f"Aerodrome: {aero}")
    if loc:   bits.append(f"Location: {loc}")
    if prov:  bits.append(f"Province: {prov}")
    if narr:  bits.append(f"Narrative: {narr}")
    return title + "\n" + "\n".join(bits)

# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Import CADORS CSV into sreports")
    p.add_argument("--csv", required=True, help="CSV path (UTF-8)")
    p.add_argument("--embed", action="store_true", help="Generate OpenAI embeddings (costs tokens)")
    p.add_argument("--max-rows", type=int, default=None, help="Ingest at most N rows")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr); sys.exit(1)

    if not args.embed:
        print("[INFO] running WITHOUT embeddings (use --embed to enable)", file=sys.stderr)

    init_tables()

    total = inserted = skipped = dup = 0
    for row in smart_rows(args.csv):
        total += 1
        if args.max_rows and inserted + skipped + dup >= args.max_rows:
            break

        cad_no = safe(row, "Cadors Number")
        if not cad_no:
            skipped += 1
            continue

        if cadors_exists(cad_no):
            dup += 1
            continue

        try:
            text = build_report_text(row)
            emb = get_embedding(text) if args.embed else None
            rid = insert_sreport("Imported (CADORS)", "English", text, emb)
            link_cadors(cad_no, rid)
            inserted += 1
        except Exception as e:
            skipped += 1
            print(f"[WARN] insert failed (CADORS {cad_no}): {e}", file=sys.stderr)

    print(f"DONE. total={total} inserted={inserted} skipped={skipped} dup={dup}")

if __name__ == "__main__":
    main()
