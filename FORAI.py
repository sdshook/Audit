#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
FORAI.py (c) 2025 All Rights Reserved Shane D. Shook
Automated collection and processing for essential forensic Q&A
Supported by TinyLLaMA 1.1b
Note: prototype utilizing KAPE and Eric Zimmerman's Tools
requirements (pip install pandas wmi pywin32 fpdf llama-cpp-python)
dotNet 9 performs better than 6 also...

Directory layout (fixed):
  D:\FORAI\
    ├─ archives/    (completed case archives)
    ├─ artifacts/   (KAPE raw collections)
    ├─ extracts/    (!EZParser CSVs + live-only supplements)
    ├─ LLM/         (reserved)
    ├─ reports/     (txt/json/pdf outputs)
    └─ tools/       (kape, etc.)

Note: This script CAN run KAPE collection & parsing (use --target-drive / --skip-collect / --skip-parse).

Run examples:
  # Live system (C:) end-to-end
  python FORAI.py --case-id CASE123 --mode ALL --target-drive C: --use-llm

  # Mounted evidence E:\ BETWEEN 07/01/2025-07/31/2025
  python FORAI.py --case-id CASE123 --mode BETWEEN --between 07012025-07312025 --target-drive E:

  # 30 days BEFORE 08/15/2025, reuse existing DB (no ingestion)
  python FORAI.py --case-id CASE123 --mode DAYS_BEFORE --target 08152025 --days 30 --no-ingest

  # Reprocess existing extracts only (skip all KAPE steps)
  python FORAI.py --case-id CASE123 --mode ALL --skip-kape --use-llm

  # Skip collection but still parse existing artifacts
  python FORAI.py --case-id CASE123 --mode ALL --skip-collect

  # Skip parsing but still collect with !SANS_Triage
  python FORAI.py --case-id CASE123 --mode ALL --skip-parse

  # Point to a different extracts folder
  python FORAI.py --case-id CASE123 --mode ALL --skip-kape --extracts-dir D:\OtherCase\extracts
  
  # Ask a custom question in a defined period
  python FORAI.py --case-id CASE123 --mode BETWEEN --between 08012025-08152025 --skip-kape --ask "Were any files exfiltrated to personal cloud storage?"

"""

import os, sys, argparse, hashlib, sqlite3, json, re, time, subprocess, shutil, zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict

try:
    import pandas as pd
except Exception:
    print("[!] pandas not found. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
    TQDM = True
except Exception:
    TQDM = False

try:
    from fpdf import FPDF
    HAVE_PDF = True
except Exception:
    HAVE_PDF = False
    
try:
    from llama_cpp import Llama
    HAVE_LLAMA = True
except Exception:
    HAVE_LLAMA = False

BASE = Path(r"D:\FORAI")
DIR_ARCHIVES = BASE / "archives"
DIR_ARTIFACTS = BASE / "artifacts"
DIR_EXTRACTS = BASE / "extracts"
DIR_LLM = BASE / "LLM"
DIR_REPORTS = BASE / "reports"
DIR_TOOLS = BASE / "tools"
DB_PATH = DIR_EXTRACTS / "forai.db"

KAPE_EXE  = DIR_TOOLS / "kape" / "kape.exe"
SQLE_MAPS = DIR_TOOLS / "kape" / "Modules" / "bin" / "SQLECmd" / "Maps"

# ---------------- LLM guardrails ----------------
GUARDRAIL_SYSTEM_PROMPT = """You are a senior digital forensics analyst.
Follow these rules strictly:
1) Ground EVERY statement only in the provided evidence. Do NOT invent or assume facts.
2) If evidence is insufficient to answer, reply: "Insufficient evidence in scope."
3) Do NOT assert crimes, motives, or malicious intent (e.g., murder, homicide, extortion, blackmail, fraud)
   unless those exact terms appear in the evidence excerpt.
4) Use neutral, non-accusatory language. Do not speculate.
5) Prefer short, factual bullets; keep narrative brief and restrained.
"""

# Terms we never want the model to introduce unless they appear verbatim in evidence.
GUARDRAIL_BANNED_TERMS = [
    r"\bmurder\b", r"\bhomicide\b", r"\bmanslaughter\b",
    r"\bextortion\b", r"\bblackmail\b", r"\bfraud\b",
    r"\bterror(ism|ist)\b", r"\bassault\b", r"\bchild\s*abuse\b",
]

_GUARDRAIL_BANNED_RE = re.compile("|".join(GUARDRAIL_BANNED_TERMS), re.IGNORECASE)

def _sanitize_against_hallucinations(text: str, evidence_blob: str) -> str:
    """
    If the answer contains banned terms that are NOT present in the evidence,
    fail closed with a neutral message.
    """
    trigger = _GUARDRAIL_BANNED_RE.search(text) is not None
    justified = _GUARDRAIL_BANNED_RE.search(evidence_blob) is not None
    if trigger and not justified:
        return ("Guardrail: Potentially defamatory or crime-specific language detected without supporting evidence. "
                "Insufficient evidence in scope.")
    return text

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evidence (
    row_id      TEXT PRIMARY KEY,
    case_id     TEXT,
    host        TEXT,
    user        TEXT,
    ts_utc      INTEGER,
    artifact    TEXT,
    src_file    TEXT,
    summary     TEXT,
    fields_json TEXT,
    src_sha256  TEXT,
    row_sha256  TEXT
);
CREATE TABLE IF NOT EXISTS sources (
    src_file     TEXT PRIMARY KEY,
    tool         TEXT,
    tool_version TEXT,
    src_sha256   TEXT,
    ingested_utc INTEGER
);
CREATE TABLE IF NOT EXISTS time_normalization_log (
    id INTEGER PRIMARY KEY,
    src_file         TEXT,
    original_ts      TEXT,
    normalized_epoch INTEGER,
    note             TEXT
);
CREATE INDEX IF NOT EXISTS ix_evidence_ts       ON evidence(ts_utc);
CREATE INDEX IF NOT EXISTS ix_evidence_artifact ON evidence(artifact);
CREATE INDEX IF NOT EXISTS ix_evidence_user     ON evidence(user);
CREATE INDEX IF NOT EXISTS ix_evidence_host     ON evidence(host);

CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
    summary, fields_json, content='', tokenize='unicode61'
);
CREATE TABLE IF NOT EXISTS evidence_fts_map (
    fts_rowid INTEGER PRIMARY KEY,
    row_id    TEXT UNIQUE
);
"""

VIEWS_SQL = {
    "mv_computer_identity": """
    CREATE TEMP VIEW mv_computer_identity AS
    SELECT DISTINCT
        host AS computer_name,
        json_extract(fields_json, '$.SystemManufacturer') AS make,
        json_extract(fields_json, '$.SystemProductName') AS model,
        json_extract(fields_json, '$.SystemSerialNumber') AS serial,
        json_extract(fields_json, '$.DriveModel') AS drive_model,
        json_extract(fields_json, '$.DriveSerial') AS drive_serial,
        ts_utc,
        src_file
    FROM evidence_scope
    WHERE artifact IN ('systeminfo', 'setupapi', 'storage');
    """,
    "mv_accounts_activity": """
    CREATE TEMP VIEW mv_accounts_activity AS
    SELECT
        user,
        json_extract(fields_json, '$.SID') AS sid,
        MIN(ts_utc) AS first_activity,
        MAX(ts_utc) AS last_activity,
        COUNT(*) AS evidence_count
    FROM evidence_scope
    GROUP BY user, sid;
    """,
    "mv_primary_user": """
    CREATE TEMP VIEW mv_primary_user AS
    SELECT
        a.user, a.sid, a.first_activity, a.last_activity, a.evidence_count
    FROM mv_accounts_activity a
    WHERE a.sid IS NOT NULL
      AND a.sid LIKE 'S-1-5-21-%'               -- domain/local user SIDs
      AND (
            a.sid GLOB '*-[1-9][0-9][0-9][0-9]'  -- 4+ digit RID at end…
         OR a.sid GLOB '*-[1-9][0-9][0-9][0-9][0-9]*'  -- …or 5+ digits
      )
    ORDER BY a.last_activity DESC, a.evidence_count DESC
    LIMIT 1;
    """,
    "mv_tamper": """
    CREATE TEMP VIEW mv_tamper AS
    SELECT ts_utc, user, summary, src_file
    FROM evidence_scope
    WHERE lower(summary) LIKE '%wevtutil cl%'
       OR lower(summary) LIKE '%sdelete%'
       OR lower(summary) LIKE '%timestomp%'
       OR lower(summary) LIKE '%log cleared%';
    """,
    "mv_usb_devices": """
    CREATE TEMP VIEW mv_usb_devices AS
    SELECT
        ts_utc, user,
        json_extract(fields_json, '$.DeviceMake')  AS make,
        json_extract(fields_json, '$.DeviceModel') AS model,
        json_extract(fields_json, '$.SerialNumber') AS serial,
        src_file
    FROM evidence_scope
    WHERE artifact IN ('USBSTOR', 'setupapi', 'MountPoints2');
    """,
    "mv_usb_file_transfers": """
    CREATE TEMP VIEW mv_usb_file_transfers AS
    WITH base AS (
      SELECT
          ts_utc,
          user,
          artifact,
          summary,
          src_file,
          -- unify the file path across tools
          COALESCE(
            json_extract(fields_json,'$.FullPath'),
            json_extract(fields_json,'$.TargetPath'),
            json_extract(fields_json,'$.FileName')
          ) AS file_path,
          -- unify action/operation across tools
          lower(COALESCE(
            json_extract(fields_json,'$.Reason'),
            json_extract(fields_json,'$.UsnReason'),
            json_extract(fields_json,'$.Operation')
          )) AS action,
          -- common device/drive hints (when provided by EZ_Parser outputs)
          json_extract(fields_json,'$.DriveType')  AS drive_type,   -- e.g., "Removable" or 2
          json_extract(fields_json,'$.BusType')    AS bus_type,     -- e.g., "USB"
          json_extract(fields_json,'$.DeviceType') AS device_type,  -- e.g., "USB"
          fields_json
      FROM evidence_scope
      WHERE artifact IN ('MFT','USNJRNL','LECmd','JumpLists')
    )
    SELECT
      ts_utc,
      user,
      file_path   AS file_name,
      action,
      src_file
    FROM base
    WHERE
      -- action heuristics for file movement/creation
      (
           action LIKE '%create%'
        OR action LIKE '%rename%'
        OR action LIKE '%write%'
        OR lower(summary) LIKE '%jumplist%'
      )
      AND
      (
          -- STRONG signals the media is removable / USB
             lower(bus_type)    = 'usb'
          OR lower(device_type) = 'usb'
          OR lower(CAST(drive_type AS TEXT)) LIKE '%removable%'
          OR (CASE
                 WHEN CAST(drive_type AS TEXT) GLOB '[0-9]*'
                 THEN CAST(drive_type AS INTEGER)
                 ELSE NULL
              END) = 2   -- Windows DRIVE_REMOVABLE

          -- FALLBACK signals (useful when tools don't emit typed fields):
          OR lower(summary) LIKE '%usb%'
          -- heuristic: path on a non-system letter (commonly removable). Keep conservative: exclude C:
          OR (
               file_path IS NOT NULL
               AND substr(file_path,2,1) = ':'
               AND (substr(upper(file_path),1,1) <> 'C')
             )
      );
    """,
    "mv_cloud_exfil": """
    CREATE TEMP VIEW mv_cloud_exfil AS
    SELECT
        ts_utc, user,
        COALESCE(json_extract(fields_json, '$.FileName'), json_extract(fields_json, '$.URL')) AS file_name,
        CASE
            WHEN lower(summary) LIKE '%onedrive%' OR lower(summary) LIKE '%graph.microsoft.com%' THEN 'OneDrive/SharePoint'
            WHEN lower(summary) LIKE '%dropbox%' OR lower(summary) LIKE '%content.dropboxapi.com%' THEN 'Dropbox'
            WHEN lower(summary) LIKE '%box.com%' THEN 'Box'
            WHEN lower(summary) LIKE '%drive.google%' OR lower(summary) LIKE '%www.googleapis.com/upload%' THEN 'Google Drive'
            WHEN lower(summary) LIKE '%slack%' OR lower(summary) LIKE '%slack-files.com%' THEN 'Slack'
            WHEN lower(summary) LIKE '%icloud%' THEN 'iCloud'
            ELSE 'Other'
        END AS cloud_service,
        src_file
    FROM evidence_scope
    WHERE artifact IN ('BrowserHistory','EventLog_App','FileSystem')
      AND (
        lower(summary) LIKE '%onedrive%' OR lower(summary) LIKE '%dropbox%' OR lower(summary) LIKE '%box.com%'
        OR lower(summary) LIKE '%drive.google%' OR lower(summary) LIKE '%icloud%' OR lower(summary) LIKE '%slack%'
      );
    """,
    "mv_screenshots": """
    CREATE TEMP VIEW mv_screenshots AS
    SELECT
        ts_utc, user,
        json_extract(fields_json, '$.FilePath') AS screenshot_file,
        src_file
    FROM evidence_scope
    WHERE artifact IN ('FileSystem')
      AND (
        lower(summary) LIKE '%.png%' OR lower(summary) LIKE '%.jpg%' OR lower(summary) LIKE '%screenshot%'
      );
    """,
    "mv_printing": """
    CREATE TEMP VIEW mv_printing AS
    SELECT
        ts_utc, user,
        json_extract(fields_json, '$.DocumentName') AS document,
        json_extract(fields_json, '$.PrinterName')  AS printer,
        src_file
    FROM evidence_scope
    WHERE artifact IN ('PrintService', 'Spooler');
    """,
    "mv_installs_services": """
    CREATE TEMP VIEW mv_installs_services AS
    SELECT
        ts_utc, user,
        json_extract(fields_json, '$.ProgramName') AS program,
        json_extract(fields_json, '$.ServiceName') AS service,
        summary, src_file
    FROM evidence_scope
    WHERE artifact IN ('setupapi', 'Amcache', 'Services', 'EventLog_App');
    """,
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode('utf-8', errors='ignore')).hexdigest()

ARTIFACT_HINTS = [
    (re.compile(r"systeminfo", re.I),        "systeminfo"),
    (re.compile(r"setupapi",   re.I),        "setupapi"),
    (re.compile(r"storage|disk|physicaldrive", re.I), "storage"),
    (re.compile(r"sam",        re.I),        "SAM"),
    (re.compile(r"ntuser",     re.I),        "NTUSER"),
    (re.compile(r"event.*logon|logon|4624|4647|4634", re.I), "EventLog_Logon"),
    (re.compile(r"usbstor",    re.I),        "USBSTOR"),
    (re.compile(r"mountpoints2", re.I),      "MountPoints2"),
    (re.compile(r"mfte?cmd.*\$(?:mft|j)|\$(?:mft)\b", re.I), "MFT"),
    (re.compile(r"usnjrnl|\$j", re.I),      "USNJRNL"),
    (re.compile(r"lecmd|jumplist", re.I),    "LECmd"),
    (re.compile(r"browser|history|edge|chrome|firefox", re.I), "BrowserHistory"),
    (re.compile(r"dns", re.I),                "DNSCache"),
    (re.compile(r"process|pslist|tasklist", re.I), "Process"),
    (re.compile(r"recentdocs|shellbags|filesystem", re.I), "FileSystem"),
    (re.compile(r"printservice|spool", re.I), "PrintService"),
    (re.compile(r"spool", re.I),              "Spooler"),
    (re.compile(r"amcache", re.I),            "Amcache"),
    (re.compile(r"services", re.I),           "Services"),
    (re.compile(r"event.*app|application", re.I), "EventLog_App"),
]

KNOWN_TIME_COLS = [
    "TimeCreated","EventCreatedTime","Timestamp","TimeStamp",
    "Created","CreationTime","LastAccess","LastWrite","LastWriteTime","FirstRun","LastRun",
    "RecordCreateTime","FileCreated","FileModified","Modified","WriteTime","Accessed"
]

def parse_mmddyyyy(s: str) -> datetime:
    return datetime.strptime(s, "%m%d%Y").replace(tzinfo=timezone.utc)

def compute_range(mode: str, between: Optional[str], target: Optional[str], days: Optional[int], con: sqlite3.Connection) -> Tuple[int,int,str]:
    """
    Return (start_epoch, end_epoch, human_text).
    - For ALL with an empty DB, return (now, now, '(no evidence ingested yet — empty scope)').
    - BETWEEN and DAYS_BEFORE behave the same regardless of DB contents.
    """
    cur = con.execute("SELECT MIN(ts_utc), MAX(ts_utc) FROM evidence")
    obs_min, obs_max = cur.fetchone() or (None, None)

    # EMPTY DB handling for ALL
    if mode == "ALL" and (obs_min is None or obs_max is None):
        now = int(datetime.now(timezone.utc).timestamp())
        return now, now, "(no evidence ingested yet — empty scope)"

    if mode == "ALL":
        start = int(obs_min)
        end   = int(obs_max)
        text  = (
            f"{datetime.fromtimestamp(start, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} "
            f"to {datetime.fromtimestamp(end,   tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"
        )
        return start, end, text

    if mode == "BETWEEN" and between:
        m = re.match(r"^(\d{8})-(\d{8})$", between)
        if not m:
            raise ValueError("--between must be MMDDYYYY-MMDDYYYY")
        s = datetime.strptime(m.group(1), "%m%d%Y").replace(tzinfo=timezone.utc)
        e = datetime.strptime(m.group(2), "%m%d%Y").replace(tzinfo=timezone.utc) + timedelta(hours=23, minutes=59, seconds=59)
        return int(s.timestamp()), int(e.timestamp()), f"{s.strftime('%Y-%m-%d')} to {e.strftime('%Y-%m-%d')}"

    if mode == "DAYS_BEFORE" and target and days is not None:
        t = datetime.strptime(target, "%m%d%Y").replace(tzinfo=timezone.utc)
        s = t - timedelta(days=int(days))
        e = t + timedelta(hours=23, minutes=59, seconds=59)
        return int(s.timestamp()), int(e.timestamp()), f"{days} days before {t.strftime('%Y-%m-%d')}"

    raise ValueError("Invalid date mode/arguments")

def _run(cmd, cwd=None):
    print(f"\n[RUN] {' '.join(map(str, cmd))}")
    proc = subprocess.run(list(map(str, cmd)), cwd=cwd, capture_output=True, text=True, shell=False)
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")

def check_kape_prereqs():
    if not KAPE_EXE.exists():
        raise FileNotFoundError(f"KAPE not found at {KAPE_EXE}")
    if not SQLE_MAPS.exists():
        print(f"[WARN] SQLECmd Maps not found at {SQLE_MAPS}. Ensure maps are present.")

def kape_collect(tsource: str):
    _run([KAPE_EXE, '--tsource', tsource, '--tdest', DIR_ARTIFACTS, '--target', '!SANS_Triage', '--tflush'])

def kape_parse():
    _run([KAPE_EXE, '--msource', DIR_ARTIFACTS, '--mdest', DIR_EXTRACTS, '--module', '!EZParser', '--mflush'])

def run_live_only_supplements():
    stamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    reg_dir = DIR_EXTRACTS / 'Registry'
    reg_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        (['cmd.exe','/c','netstat -anob'],        f'netstat_anob_{stamp}.txt'),
        (['cmd.exe','/c','ipconfig /displaydns'], f'displaydns_{stamp}.txt'),
        (['cmd.exe','/c','tasklist /m'],          f'tasklist_m_{stamp}.txt'),
        (['cmd.exe','/c','systeminfo /fo csv'],   f'systeminfo_{stamp}.csv'),
    ]

    for cmd, fname in commands:
        print(f"[LIVE] {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, shell=False)

        # 1) Save to ARTIFACTS
        art_path = DIR_ARTIFACTS / fname
        with art_path.open('w', encoding='utf-8', errors='ignore') as f:
            if proc.stdout:
                f.write(proc.stdout)
            if proc.stderr:
                f.write('\n=== STDERR ===\n')
                f.write(proc.stderr)

        # 2) Copy to EXTRACTS\Registry
        dst_path = reg_dir / fname
        try:
            shutil.copy2(art_path, dst_path)
        except Exception:
            # Fallback: write content again if copy fails
            with dst_path.open('w', encoding='utf-8', errors='ignore') as f:
                if proc.stdout:
                    f.write(proc.stdout)
                if proc.stderr:
                    f.write('\n=== STDERR ===\n')
                    f.write(proc.stderr)

        print(f"[LIVE] saved: {art_path}  |  copied to: {dst_path}")

def copy_setupapi_logs():
    r"""
    Copy setupapi.* from artifacts\*\Windows\INF\ into extracts\Registry.
    This runs regardless of live vs. mounted evidence.
    """
    reg_dir = DIR_EXTRACTS / "Registry"
    reg_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for p in DIR_ARTIFACTS.rglob("setupapi*"):
        try:
            if not p.is_file():
                continue
            # only if the parent path contains \windows\inf\ (case-insensitive)
            parent_norm = str(p.parent).lower().replace("/", "\\")
            if "\\windows\\inf" not in parent_norm:
                continue

            dest = reg_dir / p.name
            # avoid collisions by adding a numeric suffix
            if dest.exists():
                stem, suf = p.stem, p.suffix
                i = 1
                while (reg_dir / f"{stem}_{i}{suf}").exists():
                    i += 1
                dest = reg_dir / f"{stem}_{i}{suf}"

            shutil.copy2(p, dest)
            print(f"[COPY] setupapi: {p} -> {dest}")
            copied += 1
        except Exception as e:
            print(f"[WARN] copy setupapi failed for {p}: {e}")

    print(f"[INFO] setupapi files copied: {copied}")

def detect_artifact(filename: str) -> str:
    for rx, name in ARTIFACT_HINTS:
        if rx.search(filename):
            return name
    return "Unknown"

def pick_timestamp(row: Dict[str, object], src_file: str, con: sqlite3.Connection) -> Optional[int]:
    for col in KNOWN_TIME_COLS:
        if col in row and str(row[col]).strip():
            val = str(row[col])
            for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ"):
                try:
                    dt = datetime.strptime(val, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    return int(dt.timestamp())
                except Exception:
                    pass
            if re.fullmatch(r"\d{13}", val):
                return int(int(val)//1000)
            if re.fullmatch(r"\d{10}", val):
                return int(val)
            if re.fullmatch(r"\d{16,19}", val):
                ticks = int(val)
                if ticks > 116444736000000000:
                    return int((ticks - 116444736000000000) / 10_000_000)
            try:
                dt = datetime.fromisoformat(val.replace('Z','+00:00'))
                return int(dt.astimezone(timezone.utc).timestamp())
            except Exception:
                pass
            con.execute("INSERT INTO time_normalization_log(src_file, original_ts, normalized_epoch, note) VALUES (?,?,?,?)",
                        (src_file, val, None, f"unparsed:{col}"))
    return None

def build_summary(artifact: str, row: Dict[str, object]) -> str:
    try:
        if artifact in ("EventLog_Logon",):
            eid = row.get("EventID") or row.get("Id")
            user = row.get("TargetUserName") or row.get("SubjectUserName") or row.get("User")
            comp = row.get("Computer") or row.get("Host")
            msg  = (row.get("Message") or "").replace("\n"," ")
            return f"[Logon {eid}] {user} @{comp} - {msg[:160]}"
        if artifact in ("MFT","USNJRNL"):
            op = row.get("Reason") or row.get("Operation")
            path = row.get("FullPath") or row.get("Path") or row.get("FileName")
            return f"FS {op}:{path}"
        if artifact in ("LECmd","JumpLists"):
            app = row.get("AppName") or row.get("AppId")
            tgt = row.get("TargetPath")
            return f"JumpList {app} -> {tgt}"
        if artifact in ("USBSTOR","setupapi","MountPoints2"):
            make = row.get("FriendlyName") or row.get("DeviceMake")
            serial = row.get("SerialNumber")
            return f"USB {make} {serial}"
        if artifact in ("PrintService","Spooler"):
            doc = row.get("DocumentName") or row.get("FileName")
            prn = row.get("PrinterName")
            return f"Print {doc} on {prn}"
    except Exception:
        pass
    path = row.get("Path") or row.get("TargetPath") or row.get("FullPath") or ""
    return (f"{artifact} {path}")[:200]

def ingest_extracts(con: sqlite3.Connection, case_id: str, extracts_dir: Path = DIR_EXTRACTS):
    csv_files = sorted(extracts_dir.rglob("*.csv"))
    if not csv_files:
        print(f"[!] No CSVs found under {extracts_dir}")
        return 0
    total = 0
    for csv_path in (tqdm(csv_files, desc="Ingesting") if TQDM else csv_files):
        try:
            src_hash = sha256_file(csv_path)
        except Exception:
            print(f"[!] Cannot hash {csv_path}")
            src_hash = None
        con.execute("INSERT OR REPLACE INTO sources(src_file, tool, tool_version, src_sha256, ingested_utc) VALUES (?,?,?,?,?)",
                    (str(csv_path), None, None, src_hash, int(time.time())))

        artifact = detect_artifact(csv_path.name)
        try:
            reader_iter = pd.read_csv(csv_path, header=0, low_memory=False, encoding="utf-8-sig", chunksize=20000)
            if not hasattr(reader_iter, "__iter__"):
                reader_iter = [reader_iter]
        except Exception:
            reader_iter = pd.read_csv(csv_path, header=0, low_memory=False, engine="python", chunksize=20000)
            if not hasattr(reader_iter, "__iter__"):
                reader_iter = [reader_iter]

        cur = con.cursor()
        row_counter = 0
        for df in reader_iter:
            if df is None or df.empty:
                continue
            records = df.to_dict(orient="records")
            for row in records:
                row_counter += 1
                ts = pick_timestamp(row, str(csv_path), con)
                host = row.get("Computer") or row.get("Host") or None
                user = row.get("User") or row.get("Username") or row.get("TargetUserName") or None
                row_id = sha256_text(f"{csv_path}:{row_counter}")
                fields_json = json.dumps(row)[:150000]
                summary = build_summary(artifact, row)
                cur.execute(
                    """
                    INSERT OR REPLACE INTO evidence(row_id,case_id,host,user,ts_utc,artifact,src_file,summary,fields_json,src_sha256,row_sha256)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (row_id, case_id, host, user, ts, artifact, str(csv_path), summary, fields_json, src_hash, sha256_text(f"{case_id}|{host}|{user}|{ts}|{artifact}|{summary}|{fields_json}|{src_hash}"))
                )
                # FTS upsert (prevent orphan growth on re-ingest)
                # If this row_id already exists in the map, delete the old FTS row + its map first
                cur.execute("SELECT fts_rowid FROM evidence_fts_map WHERE row_id=?", (row_id,))
                r = cur.fetchone()
                if r:
                    cur.execute("DELETE FROM evidence_fts WHERE rowid=?", (r[0],))
                    cur.execute("DELETE FROM evidence_fts_map WHERE row_id=?", (row_id,))

                # Insert fresh FTS row and map it to this row_id
                cur.execute("INSERT INTO evidence_fts(summary, fields_json) VALUES (?, ?)", (summary, fields_json))
                fts_rowid = cur.lastrowid
                cur.execute("INSERT INTO evidence_fts_map(fts_rowid, row_id) VALUES (?, ?)", (fts_rowid, row_id))
            con.commit()
        total += row_counter
    return total

def db_connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA_SQL)
    return con

def ensure_dirs():
    for d in [DIR_ARCHIVES, DIR_ARTIFACTS, DIR_EXTRACTS, DIR_LLM, DIR_REPORTS, DIR_TOOLS]:
        d.mkdir(parents=True, exist_ok=True)

def set_analysis_scope(con: sqlite3.Connection, start_epoch: int, end_epoch: int):
    # Nuke any prior scope (view or table), then materialize the scoped rows
    con.executescript("""
        DROP VIEW IF EXISTS evidence_scope;
        DROP TABLE IF EXISTS evidence_scope;
    """)
    con.execute(
        """
        CREATE TEMP TABLE evidence_scope AS
        SELECT * FROM evidence
        WHERE ts_utc BETWEEN ? AND ?
        """,
        (start_epoch, end_epoch)
    )

    # (Optional) tiny speed-ups when exploring in scope:
    # con.execute("CREATE INDEX IF NOT EXISTS ix_scope_ts   ON evidence_scope(ts_utc)")
    # con.execute("CREATE INDEX IF NOT EXISTS ix_scope_host ON evidence_scope(host)")
    # con.execute("CREATE INDEX IF NOT EXISTS ix_scope_user ON evidence_scope(user)")

    # Rebuild the temp views that reference evidence_scope
    for name, sql in VIEWS_SQL.items():
        con.execute(f"DROP VIEW IF EXISTS {name}")
        con.executescript(sql)

QUESTIONS = [
    "What is the computername?",
    "What are the Computer make / model / serialnumber?",
    "What are the Internal hard drive make / model / Windows + adapter serialnumbers?",
    "What are the UserNames, SIDs, first/last use (include built-ins)?",
    "Who is the primary user of the computer?",
    "Is there any evidence of data destruction or forensic tampering on this computer and if so, when and by what user?",
    "Have any removable storage devices been used, if so what are their Make / Model / SerialNumber and when were they used?",
    "If any removable storage devices were used, what files were copied to or copied/accessed from the storage devices, by whom, and when?",
    "Have any files been transferred to cloud storage services, if so which, by whom, and when?",
    "Were any screenshots created, if so by whom and when?",
    "Have any documents been printed, if so by whom and when, and using what printer?",
    "Have any software been installed or services been modified, if so - which, by whom and when?",
]

def get_header_values(con: sqlite3.Connection, range_text: str) -> Dict[str, str]:
    comp = None
    try:
        cur = con.execute("SELECT computer_name FROM mv_computer_identity WHERE computer_name IS NOT NULL LIMIT 1")
        row = cur.fetchone()
        if row:
            comp = row[0]
    except Exception:
        pass
    if not comp:
        cur = con.execute("SELECT host, COUNT(*) c FROM evidence_scope GROUP BY host ORDER BY c DESC LIMIT 1")
        r = cur.fetchone()
        comp = r[0] if r else None
    return {
        "ComputerName": comp or "(unknown)",
        "DateRange": range_text,
        "GeneratedUTC": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
    }

def answer_questions(con: sqlite3.Connection) -> List[Dict[str, object]]:
    results = []
    cur = con.execute("SELECT computer_name, make, model, serial, drive_model, drive_serial, ts_utc, src_file FROM mv_computer_identity ORDER BY ts_utc ASC")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 1, "question": QUESTIONS[0], "evidence": rows})
    results.append({"number": 2, "question": QUESTIONS[1], "evidence": rows})
    results.append({"number": 3, "question": QUESTIONS[2], "evidence": rows})

    cur = con.execute("SELECT user, sid, first_activity, last_activity, evidence_count FROM mv_accounts_activity ORDER BY last_activity DESC")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 4, "question": QUESTIONS[3], "evidence": rows})

    cur = con.execute("SELECT user, sid, first_activity, last_activity, evidence_count FROM mv_primary_user")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 5, "question": QUESTIONS[4], "evidence": rows})

    cur = con.execute("SELECT ts_utc, user, summary, src_file FROM mv_tamper ORDER BY ts_utc ASC")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 6, "question": QUESTIONS[5], "evidence": rows})

    cur = con.execute("SELECT ts_utc, user, make, model, serial, src_file FROM mv_usb_devices ORDER BY ts_utc ASC")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 7, "question": QUESTIONS[6], "evidence": rows})

    cur = con.execute("SELECT ts_utc, user, file_name, action, src_file FROM mv_usb_file_transfers ORDER BY ts_utc ASC")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 8, "question": QUESTIONS[7], "evidence": rows})

    cur = con.execute("SELECT ts_utc, user, file_name, cloud_service, src_file FROM mv_cloud_exfil ORDER BY ts_utc ASC")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 9, "question": QUESTIONS[8], "evidence": rows})

    cur = con.execute("SELECT ts_utc, user, screenshot_file, src_file FROM mv_screenshots ORDER BY ts_utc ASC")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 10, "question": QUESTIONS[9], "evidence": rows})

    cur = con.execute("SELECT ts_utc, user, document, printer, src_file FROM mv_printing ORDER BY ts_utc ASC")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 11, "question": QUESTIONS[10], "evidence": rows})

    cur = con.execute("SELECT ts_utc, user, program, service, summary, src_file FROM mv_installs_services ORDER BY ts_utc ASC")
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    results.append({"number": 12, "question": QUESTIONS[11], "evidence": rows})
    return results

def write_outputs(case_id: str, header: Dict[str,str], qa: List[Dict[str,object]]) -> Dict[str, str]:
    DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    json_path = DIR_REPORTS / f"FORAI_{case_id}_{ts}.json"
    with json_path.open('w', encoding='utf-8') as f:
        json.dump({"header": header, "qa": qa}, f, ensure_ascii=False, indent=2)
    txt_path = DIR_REPORTS / f"FORAI_{case_id}_{ts}.txt"
    with txt_path.open('w', encoding='utf-8', errors='ignore') as f:
        f.write(f"FORAI Report for {case_id}\n")
        for k, v in header.items():
            f.write(f"{k}: {v}\n")
        f.write("\n")
        for item in qa:
            f.write(f"Q{item['number']}. {item['question']}\n")
            ev = item['evidence']
            if not ev:
                f.write("  (no evidence in scope)\n\n")
                continue
            for row in ev[:50]:
                f.write("  - ")
                try:
                    f.write(json.dumps(row, ensure_ascii=False))
                except Exception:
                    f.write(str(row))
                f.write("\n")
            f.write("\n")
    pdf_path = ""
    if HAVE_PDF:
        try:
            pdf_path = DIR_REPORTS / f"FORAI_{case_id}_{ts}.pdf"
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"FORAI Report for {case_id}", ln=True)
            for k, v in header.items():
                pdf.cell(0, 8, f"{k}: {v}", ln=True)
            for item in qa:
                pdf.add_page()
                pdf.set_font("Arial", style='B', size=12)
                pdf.cell(0, 10, f"Q{item['number']}. {item['question']}", ln=True)
                pdf.set_font("Arial", size=10)
                ev = item['evidence']
                if not ev:
                    pdf.cell(0, 8, "(no evidence in scope)", ln=True)
                    continue
                for row in ev[:50]:
                    line = json.dumps(row, ensure_ascii=False)
                    pdf.multi_cell(0, 6, line)
            pdf.output(str(pdf_path))
        except Exception as e:
            print(f"[WARN] PDF generation failed: {e}")
            pdf_path = ""
    return {"json": str(json_path), "txt": str(txt_path), "pdf": str(pdf_path) if pdf_path else ""}

def build_llm_prompt(header: Dict[str, str], qa: List[Dict[str, object]]) -> str:
    # Concise context for a forensic executive summary
    lines = []
    lines.append("Summarize these digital forensics findings concisely, factually, and neutrally. No speculation.")
    lines.append(f"ComputerName: {header.get('ComputerName','')}")
    lines.append(f"DateRange: {header.get('DateRange','')}")
    lines.append("")
    for item in qa:
        lines.append(f"Q{item['number']}: {item['question']}")
        ev = item.get('evidence') or []
        for row in ev[:10]:  # cap context
            ts = row.get('ts_utc')
            when = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%SZ") if ts else ""
            # keep each row compact for context budget
            snippet = json.dumps(row, ensure_ascii=False)[:300]
            lines.append(f"- {when} {snippet}")
        lines.append("")
    lines.append("Rules: Only use facts present in the evidence above. "
            "If insufficient, answer exactly: Insufficient evidence in scope. "
            "Do not assert crimes/motives unless those exact terms are present in the evidence lines.")
    lines.append("Output format: bullet list of key findings, then a short 2–4 sentence narrative.")
    return "\n".join(lines)

def llm_summarize(header: Dict[str,str], qa: List[Dict[str,object]], model_path: str, max_tokens: int = 600) -> str:
    if not HAVE_LLAMA:
        raise RuntimeError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")

    llm = Llama(model_path=model_path, n_ctx=4096, chat_format="chatml")
    prompt = build_llm_prompt(header, qa)

    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.1,   # tighter
        top_p=0.9,
        stop=["</s>"]
    )
    text = resp["choices"][0]["message"]["content"].strip()
    # Build a lightweight "evidence blob" for guardrail comparison
    evidence_blob = prompt.lower()
    return _sanitize_against_hallucinations(text, evidence_blob)

def fts_search(con: sqlite3.Connection, query: str, limit: int = 200) -> List[Dict[str,object]]:
    """
    Retrieve top evidence rows matching an FTS query (contentless FTS + map join).
    """
    sql = """
    SELECT e.ts_utc, e.user, e.artifact, e.summary, e.src_file, e.fields_json
    FROM evidence_fts f
    JOIN evidence_fts_map m ON m.fts_rowid = f.rowid
    JOIN evidence e        ON e.row_id    = m.row_id
    WHERE f MATCH ?
    ORDER BY e.ts_utc DESC
    LIMIT ?
    """
    rows = []
    for r in con.execute(sql, (query, limit)):
        ts_utc, user, artifact, summary, src_file, fields_json = r
        try:
            fields = json.loads(fields_json) if fields_json else {}
        except Exception:
            fields = {"_raw_fields_json": fields_json}
        rows.append({
            "ts_utc": ts_utc,
            "user": user,
            "artifact": artifact,
            "summary": summary,
            "src_file": src_file,
            "fields": fields,
        })
    return rows

def llm_answer_custom(con: sqlite3.Connection, header: Dict[str,str], nl_question: str, model_path: str, max_tokens: int = 700) -> str:
    """
    Answer an ad-hoc natural language question using FTS to gather context,
    then ask the TinyLlama chat model to summarize strictly from that context.
    """
    if not HAVE_LLAMA:
        raise RuntimeError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")

    context_hits = fts_search(con, nl_question, limit=120)

    lines = []
    lines.append(f"QUESTION: {nl_question}")
    lines.append("")
    lines.append("EVIDENCE (most recent first):")
    for row in context_hits:
        when = datetime.utcfromtimestamp(row['ts_utc']).strftime("%Y-%m-%d %H:%M:%SZ") if row['ts_utc'] else ""
        lines.append(f"- {when} [{row['artifact']}] {row['summary']} ({row['src_file']})")

    lines.append("")
    lines.append("Instructions: Answer using ONLY the evidence above. "
                 "If insufficient, reply exactly: Insufficient evidence in scope. "
                 "Do not assert crimes/motives unless those exact terms appear verbatim in the evidence.")

    llm = Llama(model_path=model_path, n_ctx=4096, chat_format="chatml")
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
            {"role": "user",   "content": "\n".join(lines)},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
        top_p=0.9,
        stop=["</s>"]
    )
    text = resp["choices"][0]["message"]["content"].strip()
    evidence_blob = "\n".join(lines).lower()
    return _sanitize_against_hallucinations(text, evidence_blob)

def write_chain_of_custody(case_id: str) -> Path:
    DIR_REPORTS.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now().strftime('%m%d%Y')  # file name per write date
    out_path = DIR_REPORTS / f"{date_stamp}_custody.txt"

    def _iter_files(root: Path):
        for r, _, files in os.walk(root):
            for name in files:
                p = Path(r) / name
                try:
                    st = p.stat()
                    size = st.st_size
                    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')
                    h = sha256_file(p)
                    rel = str(p.relative_to(BASE))
                    yield h, size, mtime, rel
                except Exception:
                    # If a file disappears mid-walk, still note it
                    yield "ERR", -1, "N/A", str(p)

    with out_path.open('w', encoding='utf-8') as f:
        f.write("FORAI Chain of Custody Log\n")
        f.write(f"Case: {case_id}\n")
        f.write(f"Generated UTC: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}\n")
        f.write(f"Artifacts root: {DIR_ARTIFACTS}\n")
        f.write(f"Extracts root: {DIR_EXTRACTS}\n\n")

        for label, root in (("ARTIFACTS", DIR_ARTIFACTS), ("EXTRACTS", DIR_EXTRACTS)):
            f.write(f"=== {label} ===\n")
            count = 0
            for h, size, mtime, rel in _iter_files(root):
                f.write(f"{mtime}  {size:>12}  {h}  {rel}\n")
                count += 1
            f.write(f"(total files: {count})\n\n")

    return out_path

def write_case_archive() -> Path:
    """
    Create archives\\MMDDYYYY.zip containing artifacts\\, extracts\\, and reports\\ trees.
    If a same-day archive already exists, append _HHMMSS to avoid overwriting.
    """
    DIR_ARCHIVES.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now().strftime('%m%d%Y')
    out_zip = DIR_ARCHIVES / f"{date_stamp}.zip"
    if out_zip.exists():
        out_zip = DIR_ARCHIVES / f"{date_stamp}_{datetime.now().strftime('%H%M%S')}.zip"

    with zipfile.ZipFile(out_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root_dir in (DIR_ARTIFACTS, DIR_EXTRACTS, DIR_REPORTS):
            if not root_dir.exists():
                continue
            for r, _, files in os.walk(root_dir):
                for name in files:
                    p = Path(r) / name
                    # store paths relative to D:\FORAI (so top-level = artifacts/, extracts/, reports/)
                    arcname = str(p.relative_to(BASE))
                    try:
                        zf.write(p, arcname)
                    except Exception as e:
                        print(f"[WARN] Skipping {p}: {e}")

    return out_zip

def main():
    global DIR_EXTRACTS, DB_PATH
    
    parser = argparse.ArgumentParser(description="FORAI end-to-end analysis")
    parser.add_argument('--case-id', required=True, help='Case identifier')
    parser.add_argument('--mode', choices=['ALL','BETWEEN','DAYS_BEFORE'], default='ALL')
    parser.add_argument('--between', help='MMDDYYYY-MMDDYYYY (used with --mode BETWEEN)')
    parser.add_argument('--target', help='MMDDYYYY (used with --mode DAYS_BEFORE)')
    parser.add_argument('--days', type=int, help='Number of days before target (with --mode DAYS_BEFORE)')

    # Ingestion/processing controls
    parser.add_argument('--no-ingest', action='store_true', help='Skip CSV ingestion (reuse existing DB)')
    parser.add_argument('--extracts-dir', default=str(DIR_EXTRACTS), help='Path to existing extracts directory (default: D:\\FORAI\\extracts)')

    # KAPE controls
    parser.add_argument('--target-drive', default='C:', help='Drive letter of target (e.g., C:, E:, F:)')
    parser.add_argument('--skip-collect', action='store_true', help='Skip KAPE !SANS_Triage collection step')
    parser.add_argument('--skip-parse', action='store_true', help='Skip KAPE !EZParser parsing step')
    parser.add_argument('--skip-kape', action='store_true', help='Skip all KAPE steps (shorthand for --skip-collect --skip-parse)')
    
    # LLM Support
    parser.add_argument('--use-llm', action='store_true',
                        help='Generate an LLM executive summary from the Q&A')
    parser.add_argument('--llm-model',
                        default=str(DIR_LLM / 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'),
                        help='Path to local GGUF model (default: D:\\FORAI\\LLM\\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)')
    parser.add_argument('--llm-max-tokens', type=int, default=600,
                        help='Max tokens for LLM outputs (default: 600)')
    parser.add_argument('--ask', default=None,
                        help='Optional ad-hoc natural-language question to answer using FTS + LLM')
    
    args = parser.parse_args()

    ensure_dirs()

    if args.extracts_dir and Path(args.extracts_dir) != DIR_EXTRACTS:
        DIR_EXTRACTS = Path(args.extracts_dir)
        DB_PATH = DIR_EXTRACTS / 'forai.db'
        print(f"[INFO] Using extracts dir: {DIR_EXTRACTS}")

    try:
        if not (args.skip_kape or (args.skip_collect and args.skip_parse)):
            check_kape_prereqs()
            tdrive = args.target_drive.rstrip('\\/')
            if len(tdrive) >= 2 and tdrive[1] == ':':
                tsource = tdrive + '\\'
            else:
                raise ValueError('Invalid --target-drive (e.g., C: or E:)')
            is_live = tdrive.upper().startswith('C:')
            if not args.skip_collect:
                print(f"[INFO] Collecting with KAPE !SANS_Triage from {tsource} -> {DIR_ARTIFACTS}")
                kape_collect(tsource)
            if not args.skip_parse:
                print(f"[INFO] Parsing with KAPE !EZParser from {DIR_ARTIFACTS} -> {DIR_EXTRACTS}")
                kape_parse()
            if is_live and not args.skip_collect:
                print("[INFO] Live system detected (C:). Running live-only supplemental commands...")
                run_live_only_supplements()
            else:
                print("[INFO] Mounted evidence or KAPE skipped; not running live-only supplements.")
        else:
            print("[INFO] --skip-kape specified (or both --skip-collect and --skip-parse). Skipping all KAPE steps.")
    except Exception as e:
        print(f"[WARN] KAPE step skipped or failed: {e}")

    # Always copy setupapi.* from artifacts into extracts\Registry
    try:
        copy_setupapi_logs()
    except Exception as e:
        print(f"[WARN] setupapi copy failed: {e}")

    # Ensure user is Admin when running this program
    if os.name == 'nt':
        try:
            import ctypes
            if not bool(ctypes.windll.shell32.IsUserAnAdmin()):
                print("[i] TIP: On live C: collections, run this script as Administrator for full access.")
        except Exception:
            pass

    con = db_connect()

    if not args.no_ingest:
        print(f"[*] Ingesting CSVs from {DIR_EXTRACTS} into {DB_PATH}")
        total = ingest_extracts(con, args.case_id, DIR_EXTRACTS)
        print(f"[*] Ingested rows: {total}")
    else:
        print("[*] Skipping ingestion (using existing DB)")

    start_epoch, end_epoch, range_text = compute_range(args.mode, args.between, args.target, args.days, con)
    set_analysis_scope(con, start_epoch, end_epoch)

    header = get_header_values(con, range_text)
    qa = answer_questions(con)
    # ... after ingestion, scoping, report writes:
    out = write_outputs(args.case_id, header, qa)

    # Chain-of-custody log (artifacts + extracts)
    try:
        coc_path = write_chain_of_custody(args.case_id)
        print(f"    custody: {coc_path}")
    except Exception as e:
        print(f"[WARN] Chain-of-custody log failed: {e}")
    
    # Zip up artifacts, extracts, reports into archives\MMDDYYYY.zip
    try:
        zip_path = write_case_archive()
        print(f"    archive: {zip_path}")
    except Exception as e:
        print(f"[WARN] Case archive creation failed: {e}")

    print("[*] Outputs:")
    for k, v in out.items():
        if v:
            print(f"    {k}: {v}")

    # Optional LLM executive summary (12-question rollup)
    if args.use_llm:
        try:
            ts2 = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            summary = llm_summarize(header, qa, args.llm_model, args.llm_max_tokens)
            llm_path = DIR_REPORTS / f"FORAI_{args.case_id}_{ts2}_llm.txt"
            with llm_path.open('w', encoding='utf-8') as f:
                f.write(summary)
            print(f"    llm: {llm_path}")
        except Exception as e:
            print(f"[WARN] LLM summary failed: {e}")

    # Optional ad-hoc question answered from FTS-scoped evidence
    if args.ask:
        try:
            ts3 = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            ans = llm_answer_custom(con, header, args.ask, args.llm_model, max_tokens=max(400, args.llm_max_tokens))
            ask_path = DIR_REPORTS / f"FORAI_{args.case_id}_{ts3}_ask.txt"
            with ask_path.open('w', encoding='utf-8') as f:
                f.write("QUESTION: " + args.ask + "\n\n")
                f.write(ans)
            print(f"    ask: {ask_path}")
        except Exception as e:
            print(f"[WARN] LLM ad-hoc answer failed: {e}")

if __name__ == '__main__':
    main()
