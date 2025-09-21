# scripts/ingest_cadors.py
# CADORS (Canada) kayıtlarını CSV'den alıp mevcut "sreports" tablosuna ekler.
# "Imported (CADORS)" olarak ayarlar. app.py'ye dokunmadan çalışır.
#
# Kullanım (Railway veya lokal):
#   python scripts/ingest_cadors.py --csv data/cadors_last24m.csv --embed --max-rows 100000 --reembed --reembed-max 25000
#
# Notlar:
# - Aynı CADORS numarasını ikinci kez eklememek için "cadors_index" tablosu kullanılır.
# - sreports.method = 'Imported (CADORS)' olarak eklenir; böylece app.py içinde
#   "internal" ve "CADORS" ayrımı kolay yapılır.
# - Embedding için önce env'deki OPENAI_API_KEY'i kullanır; yoksa "config.py" dosyası varsa oradan alır.
# - --embed verilmezse embedding atlanır. --reembed ile var olan CADORS kayıtlarının boş embeddingleri sonradan doldurulur.

import os, sys, csv, json, uuid, argparse, datetime as dt
import psycopg2, psycopg2.extras

# --- OpenAI (eski 0.28 sürümü ile uyumlu) ---
try:
    import openai
except Exception as e:
    openai = None

# API key: env > config.py
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    try:
        from config import API_KEY as _K
        API_KEY = _K
    except Exception:
        API_KEY = None
if openai and API_KEY:
    try:
        openai.api_key = API_KEY
    except Exception:
        pass

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    print("ERROR: DATABASE_URL is not set.", file=sys.stderr)
    sys.exit(1)

# --------- SQL helpers ----------
def get_conn():
    return psycopg2.connect(DB_URL, sslmode="require")

def init_cadors_tables():
    """cadors_index tablosu (mapping) yoksa oluşturur."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cadors_index (
        cadors_no TEXT PRIMARY KEY,
        sreports_id UUID NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    conn.commit()
    cur.close(); conn.close()

# --------- Embedding ----------
def get_embedding(text: str):
    if (not openai) or (not API_KEY):
        raise RuntimeError("OpenAI embeddings unavailable (package or API key missing)")
    # Aynı modeli app.py'dekiyle uyumlu tutalım
    resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return resp["data"][0]["embedding"]

# --------- CSV -> metin ----------
def build_report_text(row: dict) -> str:
    """
    CADORS CSV'den anlaşılır, tek metinlik bir özet oluşturur.
    Sütun adları Transport Canada indirmesine göre değişebilir; esnek al.
    """
    def g(*keys, default=""):
        for k in keys:
            if k in row and row[k]:
                return str(row[k]).strip()
        return default

    cad_no   = g("Cadors Number", "CADORS Number", "Cadors #", default="(unknown)")
    odate    = g("Occurrence Date", "Date", default="")
    otime    = g("Occurrence Time", "Time", default="")
    otype    = g("Occurrence Type", "Type", default="")
    aerodrome= g("Aerodrome Name", "Aerodrome", default="")
    loc      = g("Occurrence Location", "Location", default="")
    prov     = g("Province", default="")
    region   = g("Occurrence Region", "Region", default="")
    event    = g("Event(s)", "Events", default="")
    cat      = g("Category(ies)", "Categories", default="")
    ac_reg   = g("Registration", default="")
    ac_make  = g("Make", default="")
    ac_model = g("Model", default="")
    phase    = g("Phase of Flight", "Phase", default="")
    narrative= g("All Narrative (Delimited by Date)", "Narrative", default="")

    header = f"[CADORS {cad_no}] {odate} {otime} — {otype}".strip()
    parts = [
        header,
        f"Aerodrome: {aerodrome} | Location: {loc} | Province/Region: {prov or '-'} / {region or '-'}",
        f"Aircraft: {ac_reg or '-'} {ac_make or ''} {ac_model or ''} | Phase: {phase or '-'}",
    ]
    if event or cat:
        parts.append(f"Events/Categories: {', '.join([p for p in [event, cat] if p])}")
    if narrative:
        parts.append(f"Narrative: {narrative}")
    return "\n".join([p for p in parts if p]).strip()

# --------- Upsert tek kayıt ----------
def upsert_cadors_row(conn, row: dict, do_embed: bool):
    cad_no = (row.get("Cadors Number") or row.get("CADORS Number") or row.get("Cadors #") or "(unknown)").strip()
    rid = uuid.uuid4()

    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # Zaten eklenmiş mi?
    cur.execute("SELECT sreports_id FROM cadors_index WHERE cadors_no=%s;", (cad_no,))
    got = cur.fetchone()
    if got:
        cur.close()
        return False, got["sreports_id"]  # skip (duplicate)

    text = build_report_text(row)

    emb = None
    if do_embed:
        try:
            emb = get_embedding(text)
        except Exception as e:
            # Embedding alınamazsa yine de kaydı metinsiz bırakmayalım
            print(f"[WARN] embedding failed for {cad_no}: {e}", file=sys.stderr)
            emb = None

    cur.execute("""
        INSERT INTO sreports (id, method, lang, report_text, result_text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s);
    """, (str(rid), "Imported (CADORS)", "English", text, "", json.dumps(emb) if emb is not None else None))
    cur.execute("INSERT INTO cadors_index (cadors_no, sreports_id) VALUES (%s, %s);", (cad_no, str(rid)))
    conn.commit()
    cur.close()
    return True, str(rid)

# --------- Geriye dönük embedding ----------
def reembed_existing(conn, limit=25000):
    """method='Imported (CADORS)' olan ve embedding'i boş olan raporlar için embedding üretir."""
    if (not openai) or (not API_KEY):
        print("[reembed] OpenAI key yok, atlanıyor.", flush=True)
        return

    cur = conn.cursor()
    cur.execute("""
        SELECT id, report_text
        FROM sreports
        WHERE method='Imported (CADORS)'
          AND (embedding IS NULL OR embedding::text='null' OR embedding::text='[]')
        ORDER BY created_at DESC
        LIMIT %s;
    """, (limit,))
    rows = cur.fetchall()
    total = len(rows)
    if total == 0:
        print("[reembed] hedef kayıt yok (embedding zaten dolu).", flush=True)
        cur.close()
        return

    print(f"[reembed] başlıyor: {total} kayıt", flush=True)
    done = 0
    for rid, txt in rows:
        try:
            emb = get_embedding(txt or "")
        except Exception as e:
            print(f"[reembed] {rid} hata: {e}", flush=True)
            continue
        cur.execute("UPDATE sreports SET embedding=%s WHERE id=%s;", (json.dumps(emb), rid))
        done += 1
        if done % 200 == 0:
            conn.commit()
            print(f"[reembed] {done}/{total}", flush=True)
    conn.commit()
    cur.close()
    print(f"[reembed] tamamlandı: {done}/{total}", flush=True)

# --------- Argparse ----------
def parse_args():
    p = argparse.ArgumentParser(description="Import CADORS CSV into sreports (with optional embeddings).")
    p.add_argument("--csv", required=True, help="Path to CADORS CSV (UTF-8/UTF-8-SIG)")
    p.add_argument("--embed", action="store_true", help="Generate embeddings while inserting")
    p.add_argument("--max-rows", type=int, default=10**9, help="Max rows to process from CSV")
    p.add_argument("--reembed", action="store_true",
                   help="Existing CADORS rows: backfill embeddings where missing.")
    p.add_argument("--reembed-max", type=int, default=25000,
                   help="How many existing CADORS rows to (re)embed at most.")
    return p.parse_args()

# --------- Main ----------
def main():
    args = parse_args()

    if args.embed and (not API_KEY or not openai):
        print("[INFO] OPENAI_API_KEY yok veya openai paketi yok. --embed devre dışı kalacak.", file=sys.stderr)
        args.embed = False
    if not args.embed:
        print("[INFO] running WITHOUT embeddings (use --embed to enable)", flush=True)

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    init_cadors_tables()

    conn = get_conn()
    total, inserted, skipped, dup = 0, 0, 0, 0

    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total > args.max_rows:
                break
            try:
                ok, _rid = upsert_cadors_row(conn, row, do_embed=args.embed)
                if ok:
                    inserted += 1
                else:
                    dup += 1
            except Exception as e:
                skipped += 1
                print(f"[WARN] insert fail (row #{total}): {e}", file=sys.stderr)

    print(f"DONE. total={total} inserted={inserted} skipped={skipped} dup={dup}", flush=True)

    # İstenirse mevcut CADORS kayıtlarının boş embeddinglerini doldur
    if args.reembed:
        reembed_existing(conn, limit=args.reembed_max)

    conn.close()

if __name__ == "__main__":
    main()
