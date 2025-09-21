# scripts/ingest_cadors.py
# CADORS (Canada) kayıtlarını CSV'den alıp mevcut 'sreports' tablosuna
# "Imported (CADORS)" olarak aynalar. Hiçbir şekilde app.py'ye dokunmaz.
#
# Kullanım (Railway ya da lokal):
#   python scripts/ingest_cadors.py  --csv data/cadors_last24m.csv
#
# Notlar:
# - Aynı CADORS numarasını ikinci kez eklememek için 'cadors_index' tablosu kullanılır.
# - 'sreports.method' = 'Imported (CADORS)' olarak ekleriz; böylece app.py içinde
#   “Internal” ve “+CADORS” ayrımı kolay yapılır.
# - Embedding için repo’nuzdaki config.py içindeki API_KEY’i kullanır.

import os, csv, sys, json, uuid, argparse, datetime as dt
import psycopg2, psycopg2.extras
import openai
from config import API_KEY

openai.api_key = API_KEY
DB_URL = os.getenv("DATABASE_URL")

# ---- DB init (ek yardımcı tablo) --------------------------------------------
def init_cadors_tables():
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor()
    # CADORS kaydı ile sreports.id eşleştirmesi; dupe engeller
    cur.execute("""
    CREATE TABLE IF NOT EXISTS cadors_index (
        cadors_no TEXT PRIMARY KEY,
        sreports_id UUID NOT NULL,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
    conn.commit()
    cur.close(); conn.close()

# ---- OpenAI embedding -------------------------------------------------------
def get_embedding(text: str):
    # app.py ile aynı modeli kullanıyoruz
    emb = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return emb["data"][0]["embedding"]

# ---- CSV alanlarını esnekçe yakalamak --------------------------------------
def pick(row, *keys, default=""):
    for k in keys:
        if k in row and row[k]:
            return row[k]
    return default

def build_report_text(row):
    # olası kolon isimleri (CADORS export farklı şablonlarla gelebilir)
    cadors_no = pick(row, "CADORS Number", "Occurrence Number", "Occurrence_No", default="(unknown)")
    date_raw  = pick(row, "Event Date", "Occurrence Date", "Date", "Event_Date")
    loc       = pick(row, "Location", "Nearest Airport", "City/Town")
    phase     = pick(row, "Flight Phase", "Phase of Flight", "Phase")
    acft      = pick(row, "Aircraft Make/Model", "Aircraft Type", "Make/Model")
    narrative = pick(row, "Narrative", "Details", "Description", "Occurrence Summary", default="No narrative provided.")

    # tarihi normalize et
    date_str = date_raw
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            date_str = dt.datetime.strptime(date_raw, fmt).strftime("%Y-%m-%d")
            break
        except Exception:
            pass

    # App’in ‘snippet’ini zengin göstermek için düz bir metin hazırlıyoruz
    text = (
        f"CADORS #{cadors_no} | Date: {date_str} | Location: {loc} | Phase: {phase} | Aircraft: {acft}\n"
        f"{narrative.strip()}"
    )
    return cadors_no, text

# ---- Insert logic -----------------------------------------------------------
def upsert_cadors_row(cadors_no: str, report_text: str):
    """
    - cadors_index’te varsa SKIP
    - yoksa sreports’a ekle (method='Imported (CADORS)'), embedding üret
    - cadors_index’e mapping yaz
    """
    conn = psycopg2.connect(DB_URL, sslmode="require")
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Dupe kontrol
    cur.execute("SELECT sreports_id FROM cadors_index WHERE cadors_no=%s;", (cadors_no,))
    hit = cur.fetchone()
    if hit:
        cur.close(); conn.close()
        return False, str(hit["sreports_id"])

    # Insert to sreports
    emb = get_embedding(report_text)
    rid = str(uuid.uuid4())
    cur.execute("""
        INSERT INTO sreports (id, method, lang, report_text, result_text, embedding)
        VALUES (%s,%s,%s,%s,%s,%s);
    """, (rid, "Imported (CADORS)", "English", report_text, "", json.dumps(emb)))
    # Index’e kaydet
    cur.execute("INSERT INTO cadors_index (cadors_no, sreports_id) VALUES (%s,%s);", (cadors_no, rid))

    conn.commit()
    cur.close(); conn.close()
    return True, rid

# ---- CLI --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ingest CADORS CSV into sreports as mirrors.")
    parser.add_argument("--csv", required=True, help="Path to CADORS CSV file (last 24 months recommended).")
    args = parser.parse_args()

    if not DB_URL:
        print("ERROR: DATABASE_URL env yok.", file=sys.stderr)
        sys.exit(1)
    if not API_KEY:
        print("ERROR: OpenAI API_KEY yok (config.py).", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.csv):
        print(f"ERROR: CSV bulunamadı: {args.csv}", file=sys.stderr)
        sys.exit(1)

    init_cadors_tables()

    total, inserted, skipped = 0, 0, 0
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            cadors_no, text = build_report_text(row)
            try:
                ok, _rid = upsert_cadors_row(cadors_no, text)
                if ok: inserted += 1
                else:  skipped  += 1
            except Exception as e:
                # Kayıt hatası olursa devam etsin
                print(f"[WARN] {cadors_no} insert fail: {e}", file=sys.stderr)

    print(f"Done. total={total} inserted={inserted} skipped(dupe)={skipped}")

if __name__ == "__main__":
    main()
