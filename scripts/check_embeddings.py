# scripts/check_embeddings.py
# CADORS kayıtları için eksik embedding'leri doldurur (backfill) veya özet verir.
# Kullanım:
#   python scripts/check_embeddings.py --summary
#   python scripts/check_embeddings.py --backfill --batch-size 200 --limit 1000
#
# Gerekli env:
#   DATABASE_URL, OPENAI_API_KEY

import os, sys, time, argparse
import psycopg2
import psycopg2.extras as extras

# openai==0.28.1 ile uyumlu kullanım
try:
    import openai
except Exception:
    openai = None

def to_pgvector(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def get_conn():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL yok.", file=sys.stderr); sys.exit(1)
    return psycopg2.connect(db_url)

def column_exists(conn, table, col):
    with conn.cursor() as c:
        c.execute("""
            select exists (
              select 1 from information_schema.columns
              where table_name=%s and column_name=%s
            )
        """, (table, col))
        return bool(c.fetchone()[0])

def summary():
    sql = """
    with scope as (
      select id, embedding from sreports
      where method = 'Imported (CADORS)'
    )
    select
      (select count(*) from scope) as total,
      (select count(*) from scope where embedding is not null) as with_emb,
      (select count(*) from scope where embedding is null) as missing
    ;
    """
    with get_conn() as conn, conn.cursor(cursor_factory=extras.DictCursor) as cur:
        cur.execute(sql)
        r = cur.fetchone()
    print("\n--- CADORS Embedding Summary ---")
    print(f"Total CADORS rows            : {r['total']}")
    print(f"With embedding (NOT NULL)    : {r['with_emb']}")
    print(f"Missing embedding (NULL)     : {r['missing']}\n")

def backfill(batch_size=200, limit=None, model="text-embedding-3-small"):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY yok.", file=sys.stderr); sys.exit(1)
    if openai is None:
        print("ERROR: openai paketi kurulu değil (requirements.txt).", file=sys.stderr); sys.exit(1)
    openai.api_key = api_key

    fetched = 0
    total_embedded = 0
    t0 = time.time()

    while True:
        with get_conn() as conn:
            conn.autocommit = False

            # title var mı diye kontrol et
            has_title = column_exists(conn, "sreports", "title")

            select_sql = """
                select id, {text_expr} as text
                from sreports
                where method = 'Imported (CADORS)' and embedding is null
                order by created_at desc
                limit %s
            """.format(
                text_expr=(
                    "coalesce(title,'') || E'\\n' || coalesce(report_text,'')"
                    if has_title else
                    "coalesce(report_text,'')"
                )
            )

            with conn.cursor() as cur:
                cur.execute(select_sql, (batch_size,))
                rows = cur.fetchall()

                if not rows:
                    break

                for (rid, text) in rows:
                    fetched += 1
                    text = (text or "").strip()
                    # Çok aşırı uzun metinlerde gereksiz masrafı kesmek için kırp (yaklaşık 15k karakter)
                    if len(text) > 15000:
                        text = text[:15000]

                    if not text:
                        emb = [0.0] * 1536
                    else:
                        resp = openai.Embedding.create(model=model, input=[text])
                        emb = resp["data"][0]["embedding"]

                    emb_str = to_pgvector(emb)
                    cur.execute("update sreports set embedding = %s where id = %s", (emb_str, rid))
                    total_embedded += 1

            conn.commit()

        print(f"[{time.strftime('%H:%M:%S')}] batch ok: {len(rows)} | total_embedded: {total_embedded}")
        if limit is not None and total_embedded >= limit:
            break

    dt = time.time() - t0
    print(f"\nDONE. embedded={total_embedded} in {dt:.1f}s\n")
    summary()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Sadece özet ver")
    parser.add_argument("--backfill", action="store_true", help="Eksik embedding'leri doldur")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="Bu kadar kaydı işle ve çık (opsiyonel)")
    parser.add_argument("--model", type=str, default="text-embedding-3-small")
    args = parser.parse_args()

    if args.summary:
        summary(); return
    if args.backfill:
        backfill(batch_size=args.batch_size, limit=args.limit, model=args.model); return

    summary()

if __name__ == "__main__":
    main()
