# scripts/check_embeddings.py
# CADORS (method='Imported (CADORS)') kayıtları için embedding özet/backfill aracıdır.
# Kullanım:
#   python scripts/check_embeddings.py --summary
#   python scripts/check_embeddings.py --backfill --batch-size 200 --limit 1000
#   python scripts/check_embeddings.py --backfill --model text-embedding-3-large
#
# Env: DATABASE_URL, OPENAI_API_KEY
# Opsiyonel: OPENAI_EMBED_MODEL

import os, sys, time, argparse
import psycopg2
import psycopg2.extras as extras

# --- OpenAI SDK: hem v1 (OpenAI client) hem v0 (openai.Embedding) ile uyum ---
CLIENT_MODE = None  # 'v1' veya 'v0'
OpenAI = None
openai = None

try:
    from openai import OpenAI  # v1+
    CLIENT_MODE = 'v1'
except Exception:
    try:
        import openai  # v0.x
        CLIENT_MODE = 'v0'
    except Exception:
        CLIENT_MODE = None

EMBED_DIM = 1536
DEFAULT_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
MAX_INPUT_CHARS = 8000

def to_pgvector(vec):
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

def get_conn():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL yok.", file=sys.stderr); sys.exit(1)
    return psycopg2.connect(db_url)

def safe_text(title, body):
    t = (title or "").strip()
    b = (body or "").strip()
    text = (t + "\n" + b).strip()
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
    return text or "(empty)"

def mk_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY yok.", file=sys.stderr); sys.exit(1)
    if CLIENT_MODE == 'v1':
        return OpenAI(api_key=api_key)
    elif CLIENT_MODE == 'v0':
        openai.api_key = api_key
        return None
    else:
        print("ERROR: openai paketi kurulu değil. requirements.txt içine 'openai>=1.40.0' ekleyin.", file=sys.stderr)
        sys.exit(1)

def embed_text(client, model, text, retries=3, backoff=1.6):
    last = None
    for i in range(retries):
        try:
            if CLIENT_MODE == 'v1':
                resp = client.embeddings.create(model=model, input=text)
                return resp.data[0].embedding
            else:  # v0
                resp = openai.Embedding.create(model=model, input=text)
                return resp["data"][0]["embedding"]
        except Exception as e:
            last = e
            sleep = backoff ** i
            print(f"[WARN] embed hata try {i+1}/{retries}: {e} → {sleep:.1f}s", file=sys.stderr)
            time.sleep(sleep)
    raise last

SUMMARY_SQL = """
with scope as (
  select id, embedding
  from sreports
  where method = 'Imported (CADORS)'
)
select
  (select count(*) from scope)                             as total,
  (select count(*) from scope where embedding is not null) as with_emb,
  (select count(*) from scope where embedding is null)     as missing;
"""

FETCH_BATCH_SQL = """
select id, title, report_text
from sreports
where method = 'Imported (CADORS)' and embedding is null
order by created_at desc
limit %s;
"""

UPDATE_SQL = "update sreports set embedding = %s where id = %s"

def summary():
    with get_conn() as conn, conn.cursor(cursor_factory=extras.DictCursor) as cur:
        cur.execute(SUMMARY_SQL)
        r = cur.fetchone()
    print("\n--- CADORS Embedding Summary ---")
    print(f"Total CADORS rows            : {r['total']}")
    print(f"With embedding (NOT NULL)    : {r['with_emb']}")
    print(f"Missing embedding (NULL)     : {r['missing']}\n")

def backfill(batch_size=200, limit=None, model=DEFAULT_MODEL):
    client = mk_client()
    total_done = 0
    t0 = time.time()
    while True:
        with get_conn() as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute(FETCH_BATCH_SQL, (batch_size,))
                rows = cur.fetchall()
                if not rows:
                    break
                for rid, title, body in rows:
                    txt = safe_text(title, body)
                    try:
                        emb = embed_text(client, model, txt)
                    except Exception as e:
                        print(f"[ERROR] embedding yapılamadı id={rid}: {e}", file=sys.stderr)
                        continue
                    if len(emb) != EMBED_DIM:
                        print(f"[WARN] dim={len(emb)} beklenen={EMBED_DIM} (id={rid})", file=sys.stderr)
                    cur.execute(UPDATE_SQL, (to_pgvector(emb), rid))
                    total_done += 1
                    if limit is not None and total_done >= limit:
                        break
            conn.commit()
        print(f"[{time.strftime('%H:%M:%S')}] batch ok. total_embedded={total_done}")
        if limit is not None and total_done >= limit:
            break
    dt = time.time() - t0
    print(f"\nDONE. embedded={total_done} in {dt:.1f}s (model={model})\n")
    summary()

def main():
    p = argparse.ArgumentParser(description="CADORS embedding özet/backfill")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--summary", action="store_true")
    g.add_argument("--backfill", action="store_true")
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = p.parse_args()
    if args.summary or (not args.backfill):
        summary()
    else:
        backfill(batch_size=args.batch_size, limit=args.limit, model=args.model)

if __name__ == "__main__":
    main()
