import os, sys, argparse, json, time
import psycopg2, psycopg2.extras
import openai

# ---- ENV ----
DB_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
if not DB_URL:
    print("ERROR: DATABASE_URL not set"); sys.exit(1)
if not API_KEY:
    print("WARN: OPENAI_API_KEY (or API_KEY) not set. Backfill won't work.")

openai.api_key = API_KEY

def get_conn():
    return psycopg2.connect(DB_URL, sslmode="require")

def summary():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM sreports
                WHERE method LIKE 'Imported (CADORS)%';
            """)
            total = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM sreports
                WHERE method LIKE 'Imported (CADORS)%'
                  AND embedding IS NOT NULL;
            """)
            with_emb = cur.fetchone()[0]

            cur.execute("""
                SELECT COUNT(*) FROM sreports
                WHERE method LIKE 'Imported (CADORS)%'
                  AND embedding IS NULL;
            """)
            no_emb = cur.fetchone()[0]

    print("=== CADORS Embedding Summary ===")
    print(f"Total CADORS rows         : {total}")
    print(f"With embedding (NOT NULL) : {with_emb}")
    print(f"Missing embedding (NULL)  : {no_emb}")

def fetch_batch(limit):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("""
                SELECT id, report_text
                FROM sreports
                WHERE method LIKE 'Imported (CADORS)%'
                  AND (embedding IS NULL)
                ORDER BY created_at ASC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()

def compute_embedding(text):
    if not text: 
        return None
    # Keep it cheap
    resp = openai.Embedding.create(
        model="text-embedding-3-small",
        input=text[:7000]  # safety truncate
    )
    return resp["data"][0]["embedding"]

def backfill(limit, sleep_s=0.2):
    if not API_KEY:
        print("ERROR: OPENAI_API_KEY (or API_KEY) not configured."); return
    rows = fetch_batch(limit)
    if not rows:
        print("No rows to backfill."); return

    updated = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                rid = str(r["id"])
                txt = r["report_text"] or ""
                try:
                    emb = compute_embedding(txt)
                    if not emb:
                        print(f"skip (empty text): {rid}")
                        continue
                    cur.execute("UPDATE sreports SET embedding=%s WHERE id=%s;", (json.dumps(emb), rid))
                    updated += 1
                    print(f"updated: {rid}")
                    time.sleep(sleep_s)
                except Exception as e:
                    print(f"fail: {rid} -> {e}")
            conn.commit()
    print(f"BACKFILL DONE. updated={updated}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="Show counts")
    ap.add_argument("--backfill", action="store_true", help="Compute embeddings for missing CADORS rows")
    ap.add_argument("--limit", type=int, default=500, help="Max rows to backfill in one run")
    args = ap.parse_args()

    if args.summary or (not args.backfill):
        summary()
    if args.backfill:
        backfill(args.limit)
        summary()

if __name__ == "__main__":
    main()
