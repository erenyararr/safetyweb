# scripts/check_embeddings.py
# CADORS importlarından gelen (method='Imported (CADORS)') SReports kayıtları için
# embedding özetini verir veya eksik embedding'leri doldurur.
#
# Kullanım:
#   python scripts/check_embeddings.py --summary
#   python scripts/check_embeddings.py --backfill --batch-size 200 --limit 1000
#   python scripts/check_embeddings.py --backfill --model text-embedding-3-large
#
# Gerekli env: DATABASE_URL, OPENAI_API_KEY
# İsteğe bağlı env: OPENAI_EMBED_MODEL (CLI'deki --model'i override eder)

import os
import sys
import time
import math
import argparse
import psycopg2
import psycopg2.extras as extras

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ---- Helpers ---------------------------------------------------------------

EMBED_DIM = 1536  # text-embedding-3-small
DEFAULT_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
MAX_INPUT_CHARS = 8000  # güvenli truncation; token limitini aşmayalım

def to_pgvector(vec):
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"

def get_conn():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL yok.", file=sys.stderr)
        sys.exit(1)
    return psycopg2.connect(db_url)

def safe_text(title, body):
    t = (title or "").strip()
    b = (body or "").strip()
    text = (t + "\n" + b).strip()
    # truncation: çok büyük gövdeler için
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
    return text or "(empty)"

def mk_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY yok.", file=sys.stderr)
        sys.exit(1)
    if OpenAI is None:
        print("ERROR: openai paketi kurulu değil. requirements.txt → openai>=1.40.0", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key)

def embed_text(client, model, text, retries=3, backoff=1.5):
    last_err = None
    for i in range(retries):
        try:
            resp = client.embeddings.create(model=model, input=text)
            return resp.data[0].embedding
        except Exception as e:
            last_err = e
            sleep = backoff ** i
            print(f"[WARN] embed hata (try {i+1}/{retries}): {e} → {sleep:.1f}s bekle", file=sys.stderr)
            time.sleep(sleep)
    raise last_err


# ---- Queries ---------------------------------------------------------------

SUMMARY_SQL = """
with scope as (
  select id, embedding
  from sreports
  where method = 'Imported (CADORS)'
)
select
  (select count(*) from scope)                                 as total,
  (select count(*) from scope where embedding is not null)     as with_emb,
  (select count(*) from scope where embedding is null)         as missing
;
"""

FETCH_BATCH_SQL = """
select id, title, report_text
from sreports
where method = 'Imported (CADORS)' and embedding is null
order by created_at desc
limit %s
"""

UPDATE_SQL = "update sreports set embedding = %s where id = %s"


# ---- Commands --------------------------------------------------------------

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
                        # başarısız olursa sıfır vektör koyup geçmeyelim; hiç koymayıp batch sonrası deneyelim
                        print(f"[ERROR] embedding yapılamadı id={rid}: {e}", file=sys.stderr)
                        continue

                    # boyut kontrolü (model değişirse defensif)
                    if len(emb) != EMBED_DIM:
                        print(f"[WARN] dim={len(emb)} beklenen={EMBED_DIM} (id={rid})", file=sys.stderr)

                    cur.execute(UPDATE_SQL, (to_pgvector(emb), rid))
                    total_done += 1

                    if limit is not None and total_done >= limit:
                        break

            conn.commit()

        print(f"[{time.strftime('%H:%M:%S')}] batch tamamlandı, total_embedded={total_done}")

        if limit is not None and total_done >= limit:
            break

    dt = time.time() - t0
    print(f"\nDONE. embedded={total_done} in {dt:.1f}s (model={model})\n")
    summary()


# ---- CLI -------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="CADORS embedding özet/backfill")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--summary", action="store_true", help="Sadece özet ver")
    g.add_argument("--backfill", action="store_true", help="Eksik embedding'leri doldur")

    p.add_argument("--batch-size", type=int, default=200, help="Her turda kaç kayıt işlensin")
    p.add_argument("--limit", type=int, default=None, help="Toplamda en fazla kaç kayıt işlensin")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="OpenAI embedding modeli")
    args = p.parse_args()

    if args.summary:
        summary()
    elif args.backfill:
        backfill(batch_size=args.batch_size, limit=args.limit, model=args.model)
    else:
        # parametre verilmediyse özet
        summary()

if __name__ == "__main__":
    main()
