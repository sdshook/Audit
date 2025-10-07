# FORAI Case Workflow (Summary)
(c) 2025 All Rights Reserved Shane D. Shook
Note: prototype utilizing KAPE and Eric Zimmerman's Tools
requirements (pip install pandas wmi pywin32 fpdf llama-cpp-python)
dotNet 9 performs better than 6 also...

FORAI.py is an end-to-end Windows triage, analysis, and reporting pipeline intended as a prototype/POC for forensic workflows that mix deterministic methods with generative AI:

 - Acquire & Parse (optional): KAPE !SANS_Triage → artifacts\; !EZParser → extracts\; live-only supplements saved to artifacts\ and copied to extracts\Registry\.
 - Deterministic layer: CSV → SQLite (evidence, indexes, FTS); time normalization; scoped TEMP table; SQL views answering the 12 DFIR questions.
 - Generative layer (optional): LLM executive summary (--use-llm) and ad-hoc Q&A (--ask) grounded in the evidence, with strict guardrails/sanitizer.
 - Reporting & Packaging: JSON/TXT/PDF, chain-of-custody log, and daily archive ZIP.

---

## 1) Purpose

Automate forensic triage from a **live system** or **mounted evidence**, parse it, ingest into **SQLite**, scope by date, answer **12 common DFIR questions** via SQL views, and output **JSON/TXT/PDF** plus optional **LLM** summaries. Creates a **chain‑of‑custody** log and a daily **case archive ZIP**.

---

## 2) Prerequisites

* Windows host with Python 3.9+
* KAPE at `D:\FORAI\tools\kape\kape.exe` (SQLECmd Maps installed)
* Optional: `pandas`, `fpdf` (for PDF), `tqdm`, `llama-cpp-python`
* Admin privileges recommended for **live C:** collections

---

## 3) Fixed Directory Layout

```
D:\FORAI\
  ├─ artifacts\   (KAPE raw + live-only supplement outputs)
  ├─ extracts\    (!EZParser CSVs + Registry/ copies of supplements)
  ├─ reports\     (JSON/TXT/PDF + LLM summaries + MMDDYYYY_custody.txt)
  ├─ archives\    (daily case ZIPs MMDDYYYY[ _HHMMSS ].zip)
  ├─ tools\       (kape, etc.)
  └─ LLM\         (tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf although the LLM is intentionally arbitrary to valuations)
```

---

## 4) Operating Modes (date scope)

* **ALL**: Full time span observed in DB (friendly note if DB is empty)
* **BETWEEN**: `--between MMDDYYYY-MMDDYYYY`
* **DAYS\_BEFORE**: `--target MMDDYYYY --days N`

A scoped **TEMP TABLE** `evidence_scope` is materialized each run, and all analysis views select from it.

---

## 5) Acquisition & Parsing

* **Collection** (optional): KAPE target `!SANS_Triage` ⇒ `artifacts\` (from live system "C:\" or target mounted forensic images "E:\" etc.)
* **Parsing** (optional): KAPE module `!EZParser` ⇒ `extracts\`
* **Live‑only supplements (C:)**: run `netstat -anob`, `ipconfig /displaydns`, `tasklist /m`, `systeminfo /fo csv` →

  * First **saved in** `artifacts\`
  * Then **copied to** `extracts\Registry\`
* Skips:

  * `--skip-collect` (no collection)
  * `--skip-parse` (no EZParser)
  * `--skip-kape` (skip both; reprocess existing extracts)
* Reprocessing another folder: `--extracts-dir <path>`

---

## 6) Ingestion & Indexing

* Ingest all `extracts\**\*.csv` with pandas (chunked)
* SQLite tables:

  * `evidence` (row\_id, case\_id, host, user, `ts_utc`, artifact, `summary`, JSON fields, hashes)
  * `sources` (per‑CSV provenance)
  * `time_normalization_log` (unparsed timestamps)
* Indexes on time/user/host/artifact; **FTS5** over `summary` + `fields_json` (re‑ingest safe)

---

## 7) Analysis Views (12 Questions)

All views read from `evidence_scope` (the run’s date window):

1. **Computer identity** (name, make/model/serial, drive IDs)
2. **Accounts activity** (user, SID, first/last activity, counts)
3. **Primary user** (RID≥1000 heuristic, latest activity)
4. **Tampering** (log clears, sdelete, timestomp)
5. **USB devices** (make/model/serial; USBSTOR/SetupAPI/MountPoints2)
6. **USB file transfers** (MFT/USN/LECmd/JumpLists) with **removable/USB** gating
7. **Cloud exfil** (OneDrive/Google/Dropbox/Box/iCloud/Slack/Teams hints)
8. **Screenshots** (file events)
9. **Printing** (doc, printer, user, time)
10. **Installs / services** (Amcache/Services/Event logs)
    11–12. Plus any additional view‑backed questions mapped in the script

The script assembles Q→evidence lists and caps long outputs in TXT/PDF.

---

## 8) Reporting

* **JSON** and **TXT** always; **PDF** if `fpdf` is installed
* **LLM (optional)**:

  * `--use-llm` → Executive summary of the 12 Q results (`*_llm.txt`)
  * `--ask "…"` → Ad‑hoc question answered from FTS‑ranked evidence (`*_ask.txt`)
* **Guardrails**: strict system prompt (evidence‑only; say *“Insufficient evidence in scope.”* if needed) + banned‑term sanitizer (prevents crime‑word hallucinations unless present in evidence)

---

## 9) Chain of Custody & Archiving

* **Chain of custody**: `reports\MMDDYYYY_custody.txt` with recursive listings of **artifacts** + **extracts** (SHA‑256, size, UTC mtime, relative path)
* **Case archive**: `archives\MMDDYYYY.zip` (or timestamp suffix) bundling **artifacts**, **extracts**, **reports**

---

## 10) Common Commands

* Live end‑to‑end (C:):

  ```bash
  python FORAI.py --case-id CASE123 --mode ALL --target-drive C:
  ```
* Mounted evidence E: (July 2025 only):

  ```bash
  python FORAI.py --case-id CASE123 --mode BETWEEN --between 07012025-07312025 --target-drive E:
  ```
* Reprocess existing extracts only:

  ```bash
  python FORAI.py --case-id CASE123 --mode ALL --skip-kape
  ```
* Add LLM summary + ad‑hoc Q:

  ```bash
  python FORAI.py --case-id CASE123 --mode ALL --skip-kape --use-llm
  python FORAI.py --case-id CASE123 --mode ALL --skip-kape --ask "Any cloud exfil to personal storage?"
  ```

---

## 11) Operator Tips

* Run as **Administrator** for live C: collections
* If pointing `--extracts-dir` to a new folder, ensure it exists (the script will create if needed)
* For PDF, install `fpdf` (fpdf2)
* For LLM features, install `llama-cpp-python` and place the model in `D:\FORAI\LLM\`
