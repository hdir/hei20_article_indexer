"""Leser ung.no-spørsmål og -svar fra Azure SQL (tabellen dbo.qa i UngNo_QA)."""
import os
import json
import time
from typing import List

import pyodbc

# Example (PowerShell):
#   $env:UNG_AZURESQL_ODBC="Driver={ODBC Driver 18 for SQL Server};Server=tcp:<server>.database.windows.net,1433;Database=UngNo_QA;Uid=<user>;Pwd=***;Encrypt=yes;TrustServerCertificate=no;"
ODBC_CONN_STR_ENV = "UNG_AZURESQL_ODBC"

SCHEMA = "dbo"
TABLE = "qa"


def connect_mssql(timeout_sec: int = 180, max_retries: int = 10) -> pyodbc.Connection:
    cs = os.environ.get(ODBC_CONN_STR_ENV)
    if not cs:
        raise RuntimeError(f"Missing env var {ODBC_CONN_STR_ENV} with ODBC connection string")

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            cn = pyodbc.connect(cs, timeout=timeout_sec)
            cn.autocommit = False
            return cn
        except pyodbc.Error as e:
            last_err = e
            msg = str(e)

            # Azure SQL serverless can throw 40613 during resume
            if "40613" in msg or "is not currently available" in msg:
                sleep_s = min(60, 2 ** attempt)
                print(f"[db] Azure SQL resuming (40613). Retry {attempt}/{max_retries} in {sleep_s}s...")
                time.sleep(sleep_s)
                continue

            raise

    raise last_err


def fetch_questions(conn, months_back: int) -> List[dict]:
    """
    Fetch questions from the last `months_back` months that have been assigned
    a PSA/SSA sub-topic (topic_top1_psa_ssa). Rows the categorizer left as ''
    did not match any sub-topic and are filtered out.
    """
    cur = conn.cursor()
    cur.execute(f"""
    SELECT
        url,
        title,
        question,
        answer,
        topic_top1,
        topic_top1_psa_ssa,
        topics_multi_psa_ssa,
        answered_date,
        TRY_CONVERT(date, answered_date, 104) AS answered_date_dt
    FROM [{SCHEMA}].[{TABLE}]
    WHERE ok = 1
      AND question IS NOT NULL
      AND LTRIM(RTRIM(question)) <> ''
      AND topic_top1_psa_ssa IS NOT NULL
      AND topic_top1_psa_ssa <> ''
      AND answered_date IS NOT NULL
      AND TRY_CONVERT(date, answered_date, 104) IS NOT NULL
      AND TRY_CONVERT(date, answered_date, 104) >= DATEADD(month, -?, CAST(SYSUTCDATETIME() AS date))
    ORDER BY answered_date_dt ASC;
    """, (int(months_back),))

    rows = []
    for (
        url, title, question, answer,
        parent_topic, psa_ssa_top1, psa_ssa_multi_json,
        answered_raw, answered_dt,
    ) in cur.fetchall():
        try:
            psa_ssa_multi = json.loads(psa_ssa_multi_json) if psa_ssa_multi_json else []
        except (ValueError, TypeError):
            psa_ssa_multi = []

        rows.append({
            "url": url or "",
            "title": title or "",
            "question": question or "",
            "answer": answer or "",
            "parent_topic": parent_topic or "",
            "psa_ssa_topic": psa_ssa_top1 or "",
            "psa_ssa_topics_multi": psa_ssa_multi,
            "answered_date": answered_dt.isoformat() if answered_dt else (answered_raw or ""),
        })
    return rows
