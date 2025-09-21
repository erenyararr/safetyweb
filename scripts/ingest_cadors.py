#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CADORS CSV'lerini (son 24 ay vb.) alıp mevcut Postgres'teki `sreports` tablosuna
"Imported (CADORS)" method'uyla ekler ve embedding üretir.

Kullanım (Railway worker ya da lokal):
  python scripts/ingest_cadors.py --csv data/cadors_last24m.csv

Gereken ortam değişkenleri:
  - DATABASE_URL
  - OPENAI_API_KEY   (config.py içinden API_KEY fallback kabul edilir)

Notlar:
  - Aynı CADORS numarasını ikinci kez eklememek için `cadors_index` tutulur.
  - `lang` = "English", `method` = "Imported (CADORS)" yazılır.
  - `result_text` boş bırakılır (isteğe bağlı sonradan üretilebilir).
"""

import os
import sys
import csv
import json
import uuid
import argparse
import datetime as dt

import psycopg2
import psycopg2.extras
import numpy as np
import openai

# ---------- Config / Env ----------
API_KEY = os.getenv("OPENAI_API_KEY")
try:
    if not API_KEY:
        # opsiyonel: projede varsa fallback
        from config import API_KEY as _KEY_FROM_CONFIG
        API_KEY = _KEY_FROM_CONFIG
except Exception:
    pass

if not API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY. Set it in Railway Variables or provide config.API_KEY.")

openai.api_key = API_KEY

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise RuntimeError("Missing DATABASE_URL. Connect your Postgres or set the variable.")

# ---------- DB helpers ----------

def get_conn():
    return psycopg2.connect(DB_URL, sslmode="require")

def init_tables():
    """cadors_index'i oluşturur (idempotent). `sreports` zaten app.py'de var ama
    yoksa burada da oluşturmak zararsızdır."""
    ddl_cadors_index = """
    CREATE TABLE IF NOT EXISTS cadors_index (
        cadors_no TEXT PRIMARY KEY,
        sreports_id UUID NOT NULL,
        source TEXT DEFAULT 'CADORS',
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    ddl_sreports = """
    CREATE TABLE IF NOT EXISTS sreports (
        id UUID PRIMARY KEY,
        method TEXT,
        lang TEXT,
        report_text TEXT,
        result_text TEXT,
        embedding JSONB,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl_sreports)
            cur.execute(ddl_cadors_index)
        conn.commit()

def cadors_exists(cadors_no: str) -> bool:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM cadors_index WHERE cadors_no=%s;", (cadors_no,))
            return cur.fetchone() is not None

def upsert_cadors_link(cadors_no: str, report_id: str):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO cadors_index (cadors_no, sreports_id)
                VALUES (%s, %s)
                ON CONFLICT (cadors_no) DO NOTHING;
            """, (cadors_no, report_id))
        conn.commit()

# ---------- Embedding ----------
def get_embedding(text: str):
    try:
        resp = openai.Embedding.create(model="text-embedding-3-small", input=text)
        return resp["data"][0]["embedding"]
    except Exception as e:
        print(f"[WARN] Embedding failed: {e}", file=sys.stderr)
        return None

# ---------- CSV -> report_text ----------
def safe(r, key):
    v = r.get(key) if isinstance(r, dict) else None
    return (v or "").strip()

def build_report_text(r: dict) -> str:
    """
    CSV beklenen başlıklar (en azından):
      - 'Cadors Number'
      - 'Occurrence Date'
      - 'Occurrence Type'
      - 'Aerodrome Name'
      - 'Occurrence Location' (opsiyonel)
      - 'Province' (opsiyonel)
      - 'All Narrative (Delimited by Date)'  (uzun metin)
    """
    cad_no   = safe(r, "Cadors Number")
    odate    = safe(r, "Occurrence Date")
    otype    = safe(r, "Occurrence Type")
    aero     = safe(r, "Aerodrome Name")
    loc      = safe(r, "Occurrence Location")
    prov     = safe(r, "Province")
    narr     = safe(r, "All Narrative (Delimited by Date)")

    header = f"CADORS {cad_no} — {otype} on {odate} at {aero or loc or 'Unknown'} {('(' + prov + ')') if prov else ''}"
    body_lines = []
    if otype: body_lines.append(f"Occurrence Type: {otype}")
    if odate: body_lines.append(f"Date: {odate}")
    if aero:  body_lines.append(f"Aerodrome: {aero}")
    if loc:   body_lines.append(f"Location: {loc}")
    if prov:  body_lines.append(f"Province: {prov}")
    if narr:  body_lines.append(f"Narrative: {narr}")

    return header + "\n" + "\n".join(body_lines)

# ---------- Insert ----------
def insert_sreport(method: str, lang: str, report_text: str, embedding) -> str:
    rid = str(uuid.uuid4())
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sreports (id, method, lang, report_text, result_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s);
            """, (rid, method, lang, report_text, "", json.dumps(embedding) if embedding is not None else None))
        conn.commit()
    return rid

# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(description="Import CADORS CSV into sreports.")
    parser.add_argument("--csv", required=True, help="Path to CADORS CSV file (UTF-8).")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    init_tables()

    total = inserted = skipped = dup = 0
    with open(args.csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            cad_no = safe(row, "Cadors Number")
            if not cad_no:
                skipped += 1
                continue

            if cadors_exists(cad_no):
                dup += 1
                continue

            try:
                text = build_report_text(row)
                emb  = get_embedding(text)
                rid  = insert_sreport("Imported (CADORS)", "English", text, emb)
                upsert_cadors_link(cad_no, rid)
                inserted += 1
            except Exception as e:
                skipped += 1
                print(f"[WARN] Row insert failed (CADORS {cad_no}): {e}", file=sys.stderr)

    print(f"DONE. total={total} inserted={inserted} skipped={skipped} dup={dup}")

if __name__ == "__main__":
    main()
