#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New_FORAI.py (c) 2025 All Rights Reserved Shane D. Shook
Refactored for enhanced accuracy, completeness, and efficiency
Automated collection and processing for essential forensic Q&A
Supported by TinyLLaMA 1.1b with enhanced guardrails

IMPROVEMENTS IN THIS REFACTOR:
- Enhanced error handling and logging
- Improved memory management and performance
- Better configuration management
- Enhanced LLM guardrails and accuracy
- Optimized database operations
- Improved parallel processing
- Better code organization and documentation
- Enhanced data validation and sanitization

=============Order of Script Definition======================================
 IMPORTS - All external dependencies with better error handling
 CONFIGURATION - Centralized configuration management
 LOGGING - Enhanced logging system
 GLOBAL VARIABLES AND CACHES - Performance optimization state
 DATABASE SCHEMA AND VIEWS - Enhanced database structure
 UTILITY FUNCTIONS - Optimized helper functions
 TIME PROCESSING FUNCTIONS - Improved timestamp handling
 ARTIFACT DETECTION - Enhanced evidence classification
 DATABASE FUNCTIONS - Optimized database operations
 ENHANCED FTS SEARCH - Improved search accuracy
 LLM GUARDRAILS - Enhanced safety and accuracy
 CSV INGESTION - Optimized parallel processing
 KAPE INTEGRATION - External tool orchestration
 FORENSIC ANALYSIS - Core question answering logic
 LLM FUNCTIONS - Enhanced language model integration
 OUTPUT AND REPORTING - Improved report generation
 MAIN FUNCTION - Enhanced workflow orchestration
=============================================================================
"""

# =============================================================================
# IMPORTS - Enhanced with better error handling
# =============================================================================

import os
import sys
import argparse
import hashlib
import sqlite3
import json
import re
import time
import subprocess
import shutil
import zipfile
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set, Union, Any, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from collections import defaultdict
from functools import lru_cache, wraps

# Optional imports with graceful degradation and better error messages
MISSING_DEPS = []

try:
    import pandas as pd
    HAVE_PANDAS = True
except ImportError:
    HAVE_PANDAS = False
    MISSING_DEPS.append("pandas")

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False
    MISSING_DEPS.append("tqdm")

try:
    from fpdf import FPDF
    HAVE_PDF = True
except ImportError:
    HAVE_PDF = False
    MISSING_DEPS.append("fpdf")
    
try:
    from llama_cpp import Llama
    HAVE_LLAMA = True
except ImportError:
    HAVE_LLAMA = False
    MISSING_DEPS.append("llama-cpp-python")

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    MISSING_DEPS.append("psutil")

# =============================================================================
# CONFIGURATION - Centralized configuration management
# =============================================================================

@dataclass
class ForaiConfig:
    """Centralized configuration management for FORAI"""
    
    # Directory structure
    base_dir: Path = Path(r"D:\FORAI")
    
    @property
    def dir_archives(self) -> Path:
        return self.base_dir / "archives"
    
    @property
    def dir_artifacts(self) -> Path:
        return self.base_dir / "artifacts"
    
    @property
    def dir_extracts(self) -> Path:
        return self.base_dir / "extracts"
    
    @property
    def dir_llm(self) -> Path:
        return self.base_dir / "LLM"
    
    @property
    def dir_reports(self) -> Path:
        return self.base_dir / "reports"
    
    @property
    def dir_tools(self) -> Path:
        return self.base_dir / "tools"
    
    @property
    def db_path(self) -> Path:
        return self.dir_extracts / "forai.db"
    
    # External tool paths
    @property
    def kape_exe(self) -> Path:
        return self.dir_tools / "kape" / "kape.exe"
    
    @property
    def sqle_maps(self) -> Path:
        return self.dir_tools / "kape" / "Modules" / "bin" / "SQLECmd" / "Maps"
    
    # Performance settings
    max_workers: int = 4
    batch_size: int = 5000
    chunk_size: int = 10000
    max_memory_usage: float = 0.8  # 80% of available memory
    
    # LLM settings
    llm_context_size: int = 8192
    llm_max_tokens: int = 1200
    llm_temperature: float = 0.05
    llm_top_p: float = 0.85
    llm_threads: int = 4
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        if not self.base_dir.exists():
            issues.append(f"Base directory does not exist: {self.base_dir}")
        
        if not self.kape_exe.exists():
            issues.append(f"KAPE executable not found: {self.kape_exe}")
        
        if self.max_workers < 1:
            issues.append("max_workers must be at least 1")
        
        if self.batch_size < 100:
            issues.append("batch_size should be at least 100 for efficiency")
        
        return issues

# Global configuration instance
CONFIG = ForaiConfig()

# =============================================================================
# LOGGING - Enhanced logging system
# =============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Setup enhanced logging system"""
    
    # Create logger
    logger = logging.getLogger("FORAI")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Initialize logger
LOGGER = setup_logging()

# =============================================================================
# GLOBAL VARIABLES AND CACHES - Enhanced performance optimization
# =============================================================================

# LLM guardrails configuration
GUARDRAIL_SYSTEM_PROMPT = """You are a senior digital forensics analyst with expertise in evidence analysis.
Follow these rules strictly for maximum accuracy:

1) EVIDENCE-BASED ANALYSIS: Ground EVERY statement only in the provided evidence. 
   Do NOT invent, assume, or extrapolate facts beyond what is explicitly present.

2) INSUFFICIENT EVIDENCE: If evidence is insufficient to answer a question, 
   reply exactly: "Insufficient evidence in scope."

3) NEUTRAL LANGUAGE: Do NOT assert crimes, motives, or malicious intent 
   (e.g., murder, homicide, extortion, blackmail, fraud) unless those exact 
   terms appear in the evidence excerpt.

4) FACTUAL REPORTING: Use neutral, non-accusatory language. Avoid speculation.
   Report what happened, when, and by whom based solely on evidence.

5) STRUCTURED OUTPUT: Prefer short, factual bullets. Keep narrative brief and restrained.
   Include specific timestamps, file names, and user accounts when available.

6) LIMITATIONS: Always note any limitations in the available evidence.
   Distinguish between correlation and causation.
"""

GUARDRAIL_BANNED_TERMS = [
    r"\bmurder\b", r"\bhomicide\b", r"\bmanslaughter\b",
    r"\bextortion\b", r"\bblackmail\b", r"\bfraud\b",
    r"\bterror(ism|ist)\b", r"\bassault\b", r"\bchild\s*abuse\b",
    r"\bkill(ed|ing)?\b", r"\bstalk(ed|ing)?\b", r"\bharas(s|sed|sing)\b"
]

# Compiled regex for performance
_GUARDRAIL_BANNED_RE = re.compile("|".join(GUARDRAIL_BANNED_TERMS), re.IGNORECASE)

# Enhanced artifact detection patterns
ARTIFACT_HINTS = [
    (re.compile(r"systeminfo", re.I), "systeminfo"),
    (re.compile(r"setupapi", re.I), "setupapi"),
    (re.compile(r"storage|disk|physicaldrive", re.I), "storage"),
    (re.compile(r"sam", re.I), "SAM"),
    (re.compile(r"ntuser", re.I), "NTUSER"),
    (re.compile(r"event.*logon|logon|4624|4647|4634", re.I), "EventLog_Logon"),
    (re.compile(r"usbstor", re.I), "USBSTOR"),
    (re.compile(r"mountpoints2", re.I), "MountPoints2"),
    (re.compile(r"mfte?cmd.*\$(?:mft|j)|\$(?:mft)\b", re.I), "MFT"),
    (re.compile(r"usnjrnl|\$j", re.I), "USNJRNL"),
    (re.compile(r"lecmd|jumplist", re.I), "LECmd"),
    (re.compile(r"browser|history|edge|chrome|firefox", re.I), "BrowserHistory"),
    (re.compile(r"dns", re.I), "DNSCache"),
    (re.compile(r"process|pslist|tasklist", re.I), "Process"),
    (re.compile(r"recentdocs|shellbags|filesystem", re.I), "FileSystem"),
    (re.compile(r"printservice|spool", re.I), "PrintService"),
    (re.compile(r"amcache", re.I), "Amcache"),
    (re.compile(r"services", re.I), "Services"),
    (re.compile(r"event.*app|application", re.I), "EventLog_App"),
    (re.compile(r"prefetch", re.I), "Prefetch"),
    (re.compile(r"registry|reg", re.I), "Registry"),
    (re.compile(r"network|netstat", re.I), "Network"),
]

# Enhanced timestamp parsing configuration
KNOWN_TIME_COLS = [
    "TimeCreated", "EventCreatedTime", "Timestamp", "TimeStamp",
    "Created", "CreationTime", "LastAccess", "LastWrite", "LastWriteTime", 
    "FirstRun", "LastRun", "RecordCreateTime", "FileCreated", "FileModified", 
    "Modified", "WriteTime", "Accessed", "AccessedTime", "ModifiedTime",
    "ExecutionTime", "InstallTime", "UninstallTime", "StartTime", "EndTime"
]

# Enhanced forensic questions template
FORENSIC_QUESTIONS = [
    "What is the computername and system information?",
    "What are the Computer make, model, and serial number?",
    "What are the Internal hard drive make, model, Windows version, and adapter serial numbers?",
    "What are the UserNames, SIDs, first/last use (include built-ins and service accounts)?",
    "Who is the primary user of the computer based on activity patterns?",
    "Is there any evidence of data destruction, log clearing, or forensic tampering on this computer and if so, when and by what user?",
    "Have any removable storage devices been used, if so what are their Make, Model, Serial Number and when were they used?",
    "If any removable storage devices were used, what files were copied to or accessed from the storage devices, by whom, and when?",
    "Have any files been transferred to cloud storage services, if so which services, by whom, and when?",
    "Were any screenshots or screen recordings created, if so by whom and when?",
    "Have any documents been printed, if so by whom and when, and using what printer?",
    "Have any software been installed, uninstalled, or services been modified, if so - which, by whom and when?",
]

# Performance caches
_timestamp_format_cache = {}
_artifact_cache = {}
_connection_pool = {}
_connection_lock = threading.Lock()

# =============================================================================
# DATABASE SCHEMA AND VIEWS - Enhanced database structure
# =============================================================================

# Enhanced schema with better indexing and constraints
ENHANCED_SCHEMA_SQL = """
-- Main evidence table with enhanced constraints
CREATE TABLE IF NOT EXISTS evidence (
    row_id      TEXT PRIMARY KEY,
    case_id     TEXT NOT NULL,
    host        TEXT,
    user        TEXT,
    ts_utc      INTEGER,
    artifact    TEXT NOT NULL,
    src_file    TEXT NOT NULL,
    summary     TEXT,
    fields_json TEXT,
    src_sha256  TEXT,
    row_sha256  TEXT,
    created_utc INTEGER DEFAULT (strftime('%s', 'now')),
    CONSTRAINT fk_source FOREIGN KEY (src_file) REFERENCES sources(src_file)
);

-- Enhanced sources table
CREATE TABLE IF NOT EXISTS sources (
    src_file     TEXT PRIMARY KEY,
    tool         TEXT,
    tool_version TEXT,
    src_sha256   TEXT,
    file_size    INTEGER,
    ingested_utc INTEGER DEFAULT (strftime('%s', 'now')),
    status       TEXT DEFAULT 'processed'
);

-- Enhanced time normalization log
CREATE TABLE IF NOT EXISTS time_normalization_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    src_file         TEXT NOT NULL,
    original_ts      TEXT,
    normalized_epoch INTEGER,
    confidence       REAL DEFAULT 1.0,
    method           TEXT,
    note             TEXT,
    created_utc      INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Analysis metadata table
CREATE TABLE IF NOT EXISTS analysis_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT,
    updated_utc INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Performance-optimized indexes
CREATE INDEX IF NOT EXISTS ix_evidence_ts       ON evidence(ts_utc) WHERE ts_utc IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_evidence_artifact ON evidence(artifact);
CREATE INDEX IF NOT EXISTS ix_evidence_user     ON evidence(user) WHERE user IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_evidence_host     ON evidence(host) WHERE host IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_evidence_case     ON evidence(case_id);
CREATE INDEX IF NOT EXISTS ix_evidence_user_ts  ON evidence(user, ts_utc) WHERE user IS NOT NULL AND ts_utc IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_evidence_artifact_ts ON evidence(artifact, ts_utc) WHERE ts_utc IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_evidence_host_user ON evidence(host, user) WHERE host IS NOT NULL AND user IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_evidence_case_ts  ON evidence(case_id, ts_utc) WHERE ts_utc IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_evidence_src_file ON evidence(src_file);

-- Enhanced FTS with better tokenization
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_fts USING fts5(
    summary, fields_json, 
    content='evidence', 
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 1 categories "L* N* S*"'
);

-- Automatic FTS synchronization triggers with error handling
CREATE TRIGGER IF NOT EXISTS evidence_ai AFTER INSERT ON evidence BEGIN
  INSERT INTO evidence_fts(rowid, summary, fields_json) 
  VALUES (new.rowid, COALESCE(new.summary, ''), COALESCE(new.fields_json, '{}'));
END;

CREATE TRIGGER IF NOT EXISTS evidence_ad AFTER DELETE ON evidence BEGIN
  INSERT INTO evidence_fts(evidence_fts, rowid, summary, fields_json) 
  VALUES('delete', old.rowid, COALESCE(old.summary, ''), COALESCE(old.fields_json, '{}'));
END;

CREATE TRIGGER IF NOT EXISTS evidence_au AFTER UPDATE ON evidence BEGIN
  INSERT INTO evidence_fts(evidence_fts, rowid, summary, fields_json) 
  VALUES('delete', old.rowid, COALESCE(old.summary, ''), COALESCE(old.fields_json, '{}'));
  INSERT INTO evidence_fts(rowid, summary, fields_json) 
  VALUES (new.rowid, COALESCE(new.summary, ''), COALESCE(new.fields_json, '{}'));
END;

-- Analysis scope view (temporary table for current analysis)
CREATE TABLE IF NOT EXISTS analysis_scope (
    start_epoch INTEGER,
    end_epoch   INTEGER,
    description TEXT,
    created_utc INTEGER DEFAULT (strftime('%s', 'now'))
);
"""

# Enhanced analytical views
ENHANCED_VIEWS_SQL = {
    "evidence_norm": """
    CREATE TEMP VIEW IF NOT EXISTS evidence_norm AS
    SELECT
      e.row_id, e.case_id, e.ts_utc, e.host, e.user, e.artifact, 
      e.src_file, e.summary, e.fields_json,
      
      -- Normalized user identities with better extraction
      COALESCE(
        e.user,
        json_extract(e.fields_json,'$.User'),
        json_extract(e.fields_json,'$.UserName'),
        json_extract(e.fields_json,'$.Username'),
        json_extract(e.fields_json,'$.AccountName'),
        json_extract(e.fields_json,'$.SubjectUserName'),
        json_extract(e.fields_json,'$.TargetUserName'),
        json_extract(e.fields_json,'$.LogonUser'),
        json_extract(e.fields_json,'$.Owner')
      ) AS n_user,

      COALESCE(
        json_extract(e.fields_json,'$.SID'),
        json_extract(e.fields_json,'$.Sid'),
        json_extract(e.fields_json,'$."User SID"'),
        json_extract(e.fields_json,'$.SecurityId'),
        json_extract(e.fields_json,'$.TargetSid'),
        json_extract(e.fields_json,'$.SubjectUserSid')
      ) AS n_sid,

      -- Enhanced system identity extraction
      COALESCE(
        json_extract(e.fields_json,'$.SystemManufacturer'),
        json_extract(e.fields_json,'$.System_Manufacturer'),
        json_extract(e.fields_json,'$."System Manufacturer"'),
        json_extract(e.fields_json,'$.Manufacturer'),
        json_extract(e.fields_json,'$.Make')
      ) AS n_make,

      COALESCE(
        json_extract(e.fields_json,'$.SystemProductName'),
        json_extract(e.fields_json,'$.System_Product_Name'),
        json_extract(e.fields_json,'$."System Model"'),
        json_extract(e.fields_json,'$.Model'),
        json_extract(e.fields_json,'$.ProductName')
      ) AS n_model,

      COALESCE(
        json_extract(e.fields_json,'$.SystemSerialNumber'),
        json_extract(e.fields_json,'$.System_Serial_Number'),
        json_extract(e.fields_json,'$."Serial Number"'),
        json_extract(e.fields_json,'$.SerialNumber'),
        json_extract(e.fields_json,'$."Chassis Serial Number"')
      ) AS n_serial,

      -- Enhanced drive information
      COALESCE(
        json_extract(e.fields_json,'$.DriveModel'),
        json_extract(e.fields_json,'$.DiskModel'),
        json_extract(e.fields_json,'$.Model'),
        json_extract(e.fields_json,'$.DeviceModel')
      ) AS n_drive_model,

      COALESCE(
        json_extract(e.fields_json,'$.DriveSerial'),
        json_extract(e.fields_json,'$.DiskSerial'),
        json_extract(e.fields_json,'$.SerialNumber'),
        json_extract(e.fields_json,'$."Disk Serial Number"')
      ) AS n_drive_serial,

      -- Enhanced file path extraction
      COALESCE(
        json_extract(e.fields_json,'$.FullPath'),
        json_extract(e.fields_json,'$.TargetPath'),
        json_extract(e.fields_json,'$.FileName'),
        json_extract(e.fields_json,'$.Path'),
        json_extract(e.fields_json,'$.FilePath')
      ) AS n_file_path,

      -- Enhanced action/operation extraction
      lower(COALESCE(
        json_extract(e.fields_json,'$.Reason'),
        json_extract(e.fields_json,'$.UsnReason'),
        json_extract(e.fields_json,'$.Operation'),
        json_extract(e.fields_json,'$.Action'),
        json_extract(e.fields_json,'$.EventType')
      )) AS n_action,

      -- Device type information
      json_extract(e.fields_json,'$.DriveType')  AS n_drive_type,
      json_extract(e.fields_json,'$.BusType')    AS n_bus_type,
      json_extract(e.fields_json,'$.DeviceType') AS n_device_type,

      -- Lowercase for efficient searching
      lower(e.summary)     AS lsum,
      lower(e.fields_json) AS ljson
      
    FROM evidence e
    WHERE EXISTS (SELECT 1 FROM analysis_scope) 
      AND (e.ts_utc IS NULL OR e.ts_utc BETWEEN 
           (SELECT start_epoch FROM analysis_scope LIMIT 1) AND 
           (SELECT end_epoch FROM analysis_scope LIMIT 1));
    """,

    "mv_computer_identity": """
    CREATE TEMP VIEW IF NOT EXISTS mv_computer_identity AS
    SELECT DISTINCT
      host AS computer_name,
      n_make   AS make,
      n_model  AS model,
      n_serial AS serial,
      n_drive_model  AS drive_model,
      n_drive_serial AS drive_serial,
      ts_utc,
      src_file,
      COUNT(*) OVER (PARTITION BY host, n_make, n_model) as confidence_score
    FROM evidence_norm
    WHERE artifact IN ('systeminfo','setupapi','storage','Registry')
      AND (n_make IS NOT NULL OR n_model IS NOT NULL OR n_serial IS NOT NULL)
    ORDER BY confidence_score DESC, ts_utc DESC;
    """,

    "mv_accounts_activity": """
    CREATE TEMP VIEW IF NOT EXISTS mv_accounts_activity AS
    SELECT
      n_user AS user,
      n_sid  AS sid,
      MIN(ts_utc) AS first_activity,
      MAX(ts_utc) AS last_activity,
      COUNT(*)    AS evidence_count,
      COUNT(DISTINCT artifact) AS artifact_types,
      -- Classify account type
      CASE 
        WHEN n_sid LIKE 'S-1-5-21-%' THEN 'domain_user'
        WHEN n_sid LIKE 'S-1-5-18' THEN 'system'
        WHEN n_sid LIKE 'S-1-5-19' THEN 'local_service'
        WHEN n_sid LIKE 'S-1-5-20' THEN 'network_service'
        WHEN n_sid LIKE 'S-1-5-%' THEN 'builtin'
        ELSE 'unknown'
      END AS account_type
    FROM evidence_norm
    WHERE n_user IS NOT NULL
    GROUP BY n_user, n_sid;
    """,

    "mv_primary_user": """
    CREATE TEMP VIEW IF NOT EXISTS mv_primary_user AS
    SELECT
      a.user, a.sid, a.first_activity, a.last_activity, 
      a.evidence_count, a.artifact_types, a.account_type
    FROM mv_accounts_activity a
    WHERE a.account_type = 'domain_user'
      AND a.evidence_count > 10  -- Minimum activity threshold
    ORDER BY a.last_activity DESC, a.evidence_count DESC, a.artifact_types DESC
    LIMIT 1;
    """,

    "mv_tamper_evidence": """
    CREATE TEMP VIEW IF NOT EXISTS mv_tamper_evidence AS
    SELECT 
      ts_utc, n_user AS user, artifact, summary, src_file,
      CASE 
        WHEN lsum LIKE '%wevtutil%cl%' OR lsum LIKE '%clear%log%' THEN 'log_clearing'
        WHEN lsum LIKE '%sdelete%' OR lsum LIKE '%secure%delete%' THEN 'secure_deletion'
        WHEN lsum LIKE '%timestomp%' OR lsum LIKE '%timestamp%modif%' THEN 'timestamp_modification'
        WHEN lsum LIKE '%log%cleared%' OR lsum LIKE '%event%log%clear%' THEN 'event_log_clearing'
        WHEN lsum LIKE '%ccleaner%' OR lsum LIKE '%bleachbit%' THEN 'cleanup_tools'
        WHEN lsum LIKE '%cipher%/w%' THEN 'disk_wiping'
        ELSE 'potential_tampering'
      END AS tamper_type
    FROM evidence_norm
    WHERE lsum LIKE '%wevtutil%cl%'
       OR lsum LIKE '%sdelete%'
       OR lsum LIKE '%timestomp%'
       OR lsum LIKE '%log%cleared%'
       OR lsum LIKE '%clear%log%'
       OR lsum LIKE '%ccleaner%'
       OR lsum LIKE '%bleachbit%'
       OR lsum LIKE '%cipher%/w%'
    ORDER BY ts_utc DESC;
    """,

    "mv_usb_devices": """
    CREATE TEMP VIEW IF NOT EXISTS mv_usb_devices AS
    SELECT
      ts_utc,
      n_user AS user,
      COALESCE(
        json_extract(fields_json,'$.DeviceMake'),
        json_extract(fields_json,'$.FriendlyName'),
        json_extract(fields_json,'$.DeviceDesc'),
        json_extract(fields_json,'$.Product'),
        json_extract(fields_json,'$.Model'),
        json_extract(fields_json,'$.ModelName'),
        json_extract(fields_json,'$.Manufacturer')
      ) AS make,
      COALESCE(
        json_extract(fields_json,'$.DeviceModel'),
        json_extract(fields_json,'$.Model'),
        json_extract(fields_json,'$.Product'),
        json_extract(fields_json,'$.DeviceDesc')
      ) AS model,
      COALESCE(
        json_extract(fields_json,'$.SerialNumber'),
        json_extract(fields_json,'$.Serial'),
        json_extract(fields_json,'$.ContainerId'),
        json_extract(fields_json,'$.ParentIdPrefix'),
        json_extract(fields_json,'$.DeviceInstanceId')
      ) AS serial,
      artifact,
      src_file,
      -- Extract drive letter if available
      CASE 
        WHEN ljson LIKE '%driveletter%' THEN json_extract(fields_json,'$.DriveLetter')
        WHEN ljson LIKE '%mountpoint%' THEN json_extract(fields_json,'$.MountPoint')
        ELSE NULL
      END AS drive_letter
    FROM evidence_norm
    WHERE artifact IN ('USBSTOR','setupapi','MountPoints2','Registry')
      AND (ljson LIKE '%usb%' OR ljson LIKE '%removable%')
    ORDER BY ts_utc DESC;
    """
}

# =============================================================================
# UTILITY FUNCTIONS - Enhanced with better error handling
# =============================================================================

def performance_monitor(func):
    """Decorator to monitor function performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            if execution_time > 1.0:  # Log slow operations
                LOGGER.info(f"{func.__name__} took {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            LOGGER.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper

@lru_cache(maxsize=1000)
def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of file with caching"""
    try:
        hasher = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        LOGGER.error(f"Failed to hash file {path}: {e}")
        raise

@lru_cache(maxsize=10000)
def sha256_text(s: str) -> str:
    """Compute SHA256 hash of text with caching"""
    return hashlib.sha256(s.encode('utf-8')).hexdigest()

def check_memory_usage() -> Dict[str, float]:
    """Enhanced memory usage monitoring"""
    if not HAVE_PSUTIL:
        return {"available": 1.0, "used": 0.0}
    
    memory = psutil.virtual_memory()
    return {
        "total": memory.total / (1024**3),  # GB
        "available": memory.available / (1024**3),  # GB
        "used": memory.used / (1024**3),  # GB
        "percent": memory.percent / 100.0
    }

def ensure_dirs():
    """Ensure all required directories exist with proper permissions"""
    dirs_to_create = [
        CONFIG.dir_archives,
        CONFIG.dir_artifacts,
        CONFIG.dir_extracts,
        CONFIG.dir_llm,
        CONFIG.dir_reports,
        CONFIG.dir_tools
    ]
    
    for directory in dirs_to_create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            LOGGER.debug(f"Ensured directory exists: {directory}")
        except Exception as e:
            LOGGER.error(f"Failed to create directory {directory}: {e}")
            raise

def run_command(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> subprocess.CompletedProcess:
    """Enhanced command execution with better error handling"""
    try:
        LOGGER.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            cwd=cwd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            check=False
        )
        
        if result.returncode != 0:
            LOGGER.warning(f"Command failed with code {result.returncode}: {result.stderr}")
        else:
            LOGGER.debug(f"Command succeeded: {result.stdout[:200]}...")
            
        return result
    except subprocess.TimeoutExpired:
        LOGGER.error(f"Command timed out after {timeout}s: {' '.join(cmd)}")
        raise
    except Exception as e:
        LOGGER.error(f"Command execution failed: {e}")
        raise

# =============================================================================
# TIME PROCESSING FUNCTIONS - Enhanced timestamp handling
# =============================================================================

@lru_cache(maxsize=100)
def parse_mmddyyyy(s: str) -> datetime:
    """Parse MM/DD/YYYY format with caching"""
    try:
        return datetime.strptime(s, "%m/%d/%Y").replace(tzinfo=timezone.utc)
    except ValueError as e:
        LOGGER.error(f"Failed to parse date {s}: {e}")
        raise

def pick_timestamp_optimized(row: Dict[str, Any], src_file: str, con: sqlite3.Connection) -> Optional[int]:
    """Enhanced timestamp extraction with better accuracy and caching"""
    
    # Check cache first
    cache_key = f"{src_file}:{hash(str(sorted(row.items())))}"
    if cache_key in _timestamp_format_cache:
        format_info = _timestamp_format_cache[cache_key]
        if format_info:
            try:
                return int(datetime.strptime(str(row.get(format_info['column'])), format_info['format']).replace(tzinfo=timezone.utc).timestamp())
            except:
                pass
    
    # Enhanced timestamp column detection
    timestamp_candidates = []
    
    for col_name in KNOWN_TIME_COLS:
        if col_name in row and row[col_name] is not None:
            timestamp_candidates.append((col_name, row[col_name]))
    
    # Also check for any column containing 'time', 'date', or 'created'
    for col_name, value in row.items():
        if value is not None and any(keyword in col_name.lower() for keyword in ['time', 'date', 'created', 'modified']):
            if (col_name, value) not in timestamp_candidates:
                timestamp_candidates.append((col_name, value))
    
    # Try to parse each candidate
    for col_name, value in timestamp_candidates:
        try:
            parsed_time = parse_timestamp_value(value)
            if parsed_time:
                # Cache successful format
                _timestamp_format_cache[cache_key] = {
                    'column': col_name,
                    'format': detect_timestamp_format(str(value))
                }
                
                # Log normalization
                con.execute(
                    "INSERT INTO time_normalization_log(src_file, original_ts, normalized_epoch, confidence, method, note) VALUES (?,?,?,?,?,?)",
                    (src_file, str(value), int(parsed_time.timestamp()), 0.9, col_name, f"Parsed from {col_name}")
                )
                
                return int(parsed_time.timestamp())
        except Exception as e:
            LOGGER.debug(f"Failed to parse timestamp {value} from {col_name}: {e}")
            continue
    
    # Cache failure
    _timestamp_format_cache[cache_key] = None
    return None

def parse_timestamp_value(value: Any) -> Optional[datetime]:
    """Enhanced timestamp parsing with multiple format support"""
    if value is None:
        return None
    
    str_value = str(value).strip()
    if not str_value or str_value.lower() in ['null', 'none', 'n/a', '']:
        return None
    
    # Common timestamp formats to try
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y%m%d_%H%M%S",
        "%Y%m%d%H%M%S",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(str_value, fmt)
            # Assume UTC if no timezone info
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    
    # Try parsing as epoch timestamp
    try:
        timestamp = float(str_value)
        # Handle both seconds and milliseconds
        if timestamp > 1e10:  # Likely milliseconds
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (ValueError, OSError):
        pass
    
    return None

def detect_timestamp_format(value: str) -> str:
    """Detect the format of a timestamp string"""
    # This is a simplified version - in practice you'd want more sophisticated detection
    if 'T' in value and 'Z' in value:
        return "%Y-%m-%dT%H:%M:%SZ"
    elif 'T' in value:
        return "%Y-%m-%dT%H:%M:%S"
    elif '/' in value and ':' in value:
        return "%m/%d/%Y %H:%M:%S"
    elif '/' in value:
        return "%m/%d/%Y"
    elif '-' in value and ':' in value:
        return "%Y-%m-%d %H:%M:%S"
    elif '-' in value:
        return "%Y-%m-%d"
    else:
        return "%Y-%m-%d %H:%M:%S"  # Default

def compute_range(mode: str, between: Optional[str], target: Optional[str], 
                 days: Optional[int], con: sqlite3.Connection) -> Tuple[int, int, str]:
    """Enhanced time range computation with better validation"""
    
    if mode == "ALL":
        # Get actual data range from database
        cur = con.execute("SELECT MIN(ts_utc), MAX(ts_utc) FROM evidence WHERE ts_utc IS NOT NULL")
        min_ts, max_ts = cur.fetchone()
        
        if min_ts is None or max_ts is None:
            # Fallback to wide range
            start_epoch = int((datetime.now(timezone.utc) - timedelta(days=365)).timestamp())
            end_epoch = int(datetime.now(timezone.utc).timestamp())
            range_text = "ALL (no timestamps found)"
        else:
            start_epoch = min_ts
            end_epoch = max_ts
            start_dt = datetime.fromtimestamp(start_epoch, tz=timezone.utc)
            end_dt = datetime.fromtimestamp(end_epoch, tz=timezone.utc)
            range_text = f"ALL ({start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')})"
    
    elif mode == "BETWEEN":
        if not between or '-' not in between:
            raise ValueError("--between requires format MMDDYYYY-MMDDYYYY")
        
        start_str, end_str = between.split('-', 1)
        start_dt = parse_mmddyyyy(start_str)
        end_dt = parse_mmddyyyy(end_str) + timedelta(days=1) - timedelta(seconds=1)  # End of day
        
        start_epoch = int(start_dt.timestamp())
        end_epoch = int(end_dt.timestamp())
        range_text = f"BETWEEN {start_dt.strftime('%Y-%m-%d')} and {end_dt.strftime('%Y-%m-%d')}"
    
    elif mode == "DAYS_BEFORE":
        if not target or not days:
            raise ValueError("--mode DAYS_BEFORE requires both --target and --days")
        
        target_dt = parse_mmddyyyy(target)
        start_dt = target_dt - timedelta(days=days)
        end_dt = target_dt + timedelta(days=1) - timedelta(seconds=1)  # End of target day
        
        start_epoch = int(start_dt.timestamp())
        end_epoch = int(end_dt.timestamp())
        range_text = f"{days} DAYS BEFORE {target_dt.strftime('%Y-%m-%d')}"
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    LOGGER.info(f"Analysis range: {range_text}")
    return start_epoch, end_epoch, range_text

# =============================================================================
# ARTIFACT DETECTION - Enhanced evidence classification
# =============================================================================

@lru_cache(maxsize=1000)
def detect_artifact(filename: str) -> str:
    """Enhanced artifact detection with caching and better patterns"""
    
    filename_lower = filename.lower()
    
    # Check cache first
    if filename in _artifact_cache:
        return _artifact_cache[filename]
    
    # Enhanced pattern matching
    for pattern, artifact_type in ARTIFACT_HINTS:
        if pattern.search(filename_lower):
            _artifact_cache[filename] = artifact_type
            return artifact_type
    
    # Additional heuristics based on file extension and content patterns
    if filename_lower.endswith('.csv'):
        # Try to infer from filename patterns
        if 'event' in filename_lower:
            artifact_type = "EventLog"
        elif 'registry' in filename_lower or 'reg' in filename_lower:
            artifact_type = "Registry"
        elif 'file' in filename_lower:
            artifact_type = "FileSystem"
        elif 'network' in filename_lower or 'net' in filename_lower:
            artifact_type = "Network"
        else:
            artifact_type = "Unknown"
    else:
        artifact_type = "Unknown"
    
    _artifact_cache[filename] = artifact_type
    return artifact_type

def build_summary(artifact: str, row: Dict[str, Any]) -> str:
    """Enhanced summary building with better context extraction"""
    
    summary_parts = []
    
    # Artifact-specific summary building
    if artifact == "EventLog_Logon":
        event_id = row.get("EventId") or row.get("ID")
        user = (row.get("TargetUserName") or row.get("UserName") or 
                row.get("User") or row.get("Account"))
        logon_type = row.get("LogonType")
        
        if event_id == "4624":
            summary_parts.append(f"Successful logon")
        elif event_id == "4634":
            summary_parts.append(f"Logoff")
        elif event_id == "4647":
            summary_parts.append(f"User initiated logoff")
        else:
            summary_parts.append(f"Logon event {event_id}")
        
        if user:
            summary_parts.append(f"user: {user}")
        if logon_type:
            summary_parts.append(f"type: {logon_type}")
    
    elif artifact == "USBSTOR":
        device = (row.get("FriendlyName") or row.get("DeviceDesc") or 
                 row.get("Product") or row.get("Model"))
        serial = row.get("SerialNumber") or row.get("Serial")
        
        summary_parts.append("USB device")
        if device:
            summary_parts.append(f"device: {device}")
        if serial:
            summary_parts.append(f"serial: {serial}")
    
    elif artifact == "MFT":
        filename = (row.get("FileName") or row.get("Name") or 
                   row.get("FullPath") or row.get("Path"))
        action = row.get("Reason") or row.get("Operation")
        
        summary_parts.append("File system activity")
        if filename:
            summary_parts.append(f"file: {Path(filename).name}")
        if action:
            summary_parts.append(f"action: {action}")
    
    elif artifact == "BrowserHistory":
        url = row.get("URL") or row.get("Url")
        title = row.get("Title")
        
        summary_parts.append("Browser activity")
        if url:
            summary_parts.append(f"URL: {url[:50]}...")
        if title:
            summary_parts.append(f"title: {title[:30]}...")
    
    elif artifact == "Process":
        process_name = (row.get("ProcessName") or row.get("Name") or 
                       row.get("ImageName") or row.get("Process"))
        pid = row.get("PID") or row.get("ProcessId")
        
        summary_parts.append("Process activity")
        if process_name:
            summary_parts.append(f"process: {process_name}")
        if pid:
            summary_parts.append(f"PID: {pid}")
    
    else:
        # Generic summary for unknown artifacts
        # Try to find the most informative fields
        key_fields = ["Name", "FileName", "Path", "User", "Process", "Event", "Action"]
        for field in key_fields:
            if field in row and row[field]:
                summary_parts.append(f"{field.lower()}: {str(row[field])[:50]}")
                break
    
    # Fallback to first non-empty field if no summary built
    if not summary_parts:
        for key, value in row.items():
            if value and str(value).strip() and key.lower() not in ['index', 'id', 'rowid']:
                summary_parts.append(f"{key}: {str(value)[:50]}")
                break
    
    return " | ".join(summary_parts) if summary_parts else f"{artifact} record"

# =============================================================================
# DATABASE FUNCTIONS - Enhanced database operations
# =============================================================================

@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Iterator[sqlite3.Connection]:
    """Enhanced database connection with connection pooling and better configuration"""
    
    if db_path is None:
        db_path = CONFIG.db_path
    
    thread_id = threading.get_ident()
    
    with _connection_lock:
        if thread_id in _connection_pool:
            conn = _connection_pool[thread_id]
        else:
            conn = sqlite3.connect(
                str(db_path),
                timeout=30.0,
                check_same_thread=False
            )
            
            # Enhanced SQLite configuration for performance and reliability
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")  # 256MB
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Custom functions for better JSON handling
            conn.create_function("json_valid", 1, lambda x: 1 if x and json.loads(x) else 0)
            
            _connection_pool[thread_id] = conn
    
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        # Don't close - keep in pool
        pass

def db_connect() -> sqlite3.Connection:
    """Legacy function for backward compatibility"""
    conn = sqlite3.connect(str(CONFIG.db_path), timeout=30.0)
    
    # Configure SQLite for better performance
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    
    return conn

@performance_monitor
def initialize_database(con: sqlite3.Connection):
    """Enhanced database initialization with better error handling"""
    
    try:
        # Create schema
        con.executescript(ENHANCED_SCHEMA_SQL)
        
        # Create views
        for view_name, view_sql in ENHANCED_VIEWS_SQL.items():
            try:
                con.execute(view_sql)
                LOGGER.debug(f"Created view: {view_name}")
            except Exception as e:
                LOGGER.error(f"Failed to create view {view_name}: {e}")
                raise
        
        con.commit()
        LOGGER.info("Database initialized successfully")
        
    except Exception as e:
        LOGGER.error(f"Database initialization failed: {e}")
        raise

def set_analysis_scope(con: sqlite3.Connection, start_epoch: int, end_epoch: int, description: str = ""):
    """Enhanced analysis scope management"""
    
    try:
        # Clear existing scope
        con.execute("DELETE FROM analysis_scope")
        
        # Set new scope
        con.execute(
            "INSERT INTO analysis_scope(start_epoch, end_epoch, description) VALUES (?,?,?)",
            (start_epoch, end_epoch, description)
        )
        
        # Update metadata
        con.execute(
            "INSERT OR REPLACE INTO analysis_metadata(key, value) VALUES (?,?)",
            ("last_analysis_scope", f"{start_epoch}-{end_epoch}")
        )
        
        con.commit()
        LOGGER.info(f"Analysis scope set: {start_epoch} to {end_epoch}")
        
    except Exception as e:
        LOGGER.error(f"Failed to set analysis scope: {e}")
        raise

# =============================================================================
# ENHANCED FTS SEARCH - Improved search accuracy
# =============================================================================

@performance_monitor
def enhanced_fts_search(con: sqlite3.Connection, query: str, limit: int = 500) -> List[Dict[str, Any]]:
    """Enhanced full-text search with better query processing and ranking"""
    
    if not query or not query.strip():
        return []
    
    # Enhanced query preprocessing
    processed_queries = []
    
    # Original query
    processed_queries.append(query.strip())
    
    # Add quoted phrases for exact matches
    if ' ' in query:
        processed_queries.append(f'"{query.strip()}"')
    
    # Add individual terms with OR
    terms = query.strip().split()
    if len(terms) > 1:
        processed_queries.append(' OR '.join(terms))
    
    # Add wildcard searches for partial matches
    wildcard_terms = [f"{term}*" for term in terms if len(term) > 3]
    if wildcard_terms:
        processed_queries.append(' OR '.join(wildcard_terms))
    
    all_results = []
    seen_row_ids = set()
    
    for search_query in processed_queries:
        try:
            # Enhanced FTS query with ranking
            sql = """
            SELECT 
                e.row_id, e.case_id, e.host, e.user, e.ts_utc, e.artifact,
                e.src_file, e.summary, e.fields_json,
                fts.rank,
                -- Calculate relevance score
                (
                    CASE WHEN e.summary LIKE ? THEN 10 ELSE 0 END +
                    CASE WHEN e.fields_json LIKE ? THEN 5 ELSE 0 END +
                    CASE WHEN e.artifact IN ('EventLog_Logon', 'USBSTOR', 'MFT') THEN 3 ELSE 1 END
                ) as relevance_score
            FROM evidence_fts fts
            JOIN evidence e ON e.rowid = fts.rowid
            WHERE evidence_fts MATCH ?
            ORDER BY relevance_score DESC, fts.rank, e.ts_utc DESC
            LIMIT ?
            """
            
            like_pattern = f"%{query}%"
            cur = con.execute(sql, (like_pattern, like_pattern, search_query, limit))
            
            for row in cur.fetchall():
                row_dict = dict(zip([col[0] for col in cur.description], row))
                if row_dict['row_id'] not in seen_row_ids:
                    all_results.append(row_dict)
                    seen_row_ids.add(row_dict['row_id'])
                    
                    if len(all_results) >= limit:
                        break
            
            if len(all_results) >= limit:
                break
                
        except Exception as e:
            LOGGER.warning(f"FTS search failed for query '{search_query}': {e}")
            continue
    
    LOGGER.info(f"FTS search for '{query}' returned {len(all_results)} results")
    return all_results[:limit]

def build_comprehensive_context(evidence_rows: List[Dict[str, Any]], max_context_size: int = 8000) -> str:
    """Enhanced context building with better organization and relevance scoring"""
    
    if not evidence_rows:
        return "No relevant evidence found."
    
    # Group evidence by artifact type for better organization
    artifact_groups = defaultdict(list)
    for row in evidence_rows:
        artifact_groups[row.get('artifact', 'Unknown')].append(row)
    
    context_parts = []
    current_size = 0
    
    # Prioritize certain artifact types
    priority_artifacts = ['EventLog_Logon', 'USBSTOR', 'MFT', 'BrowserHistory', 'Process']
    
    # Process priority artifacts first
    for artifact in priority_artifacts:
        if artifact in artifact_groups and current_size < max_context_size:
            context_parts.append(f"\n=== {artifact} Evidence ===")
            current_size += len(context_parts[-1])
            
            for row in artifact_groups[artifact][:10]:  # Limit per artifact type
                if current_size >= max_context_size:
                    break
                
                entry = format_evidence_entry(row)
                if current_size + len(entry) < max_context_size:
                    context_parts.append(entry)
                    current_size += len(entry)
            
            del artifact_groups[artifact]
    
    # Process remaining artifacts
    for artifact, rows in artifact_groups.items():
        if current_size >= max_context_size:
            break
        
        context_parts.append(f"\n=== {artifact} Evidence ===")
        current_size += len(context_parts[-1])
        
        for row in rows[:5]:  # Fewer entries for non-priority artifacts
            if current_size >= max_context_size:
                break
            
            entry = format_evidence_entry(row)
            if current_size + len(entry) < max_context_size:
                context_parts.append(entry)
                current_size += len(entry)
    
    # Add summary statistics
    stats = f"\n=== Evidence Summary ===\nTotal evidence entries: {len(evidence_rows)}\nArtifact types: {len(set(row.get('artifact', 'Unknown') for row in evidence_rows))}\n"
    if current_size + len(stats) < max_context_size:
        context_parts.insert(0, stats)
    
    return '\n'.join(context_parts)

def format_evidence_entry(row: Dict[str, Any]) -> str:
    """Format a single evidence entry for context"""
    
    timestamp = ""
    if row.get('ts_utc'):
        try:
            dt = datetime.fromtimestamp(row['ts_utc'], tz=timezone.utc)
            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except:
            timestamp = str(row['ts_utc'])
    
    parts = []
    if timestamp:
        parts.append(f"Time: {timestamp}")
    if row.get('user'):
        parts.append(f"User: {row['user']}")
    if row.get('host'):
        parts.append(f"Host: {row['host']}")
    if row.get('summary'):
        parts.append(f"Summary: {row['summary']}")
    
    # Add relevant fields from JSON
    if row.get('fields_json'):
        try:
            fields = json.loads(row['fields_json'])
            relevant_fields = ['FileName', 'Path', 'ProcessName', 'URL', 'EventId', 'LogonType']
            for field in relevant_fields:
                if field in fields and fields[field]:
                    parts.append(f"{field}: {fields[field]}")
        except:
            pass
    
    return "- " + " | ".join(parts) + "\n"

# =============================================================================
# LLM GUARDRAILS - Enhanced safety and accuracy
# =============================================================================

def sanitize_against_hallucinations(text: str, evidence_blob: str) -> str:
    """Enhanced hallucination detection and sanitization"""
    
    if not text:
        return text
    
    # Check for banned terms
    if _GUARDRAIL_BANNED_RE.search(text):
        LOGGER.warning("Detected banned terms in LLM output, sanitizing")
        # Replace banned terms with neutral alternatives
        sanitized = _GUARDRAIL_BANNED_RE.sub("[REDACTED]", text)
        sanitized += "\n\n[NOTE: Some content was redacted due to policy restrictions]"
        return sanitized
    
    # Enhanced fact-checking against evidence
    text_lower = text.lower()
    evidence_lower = evidence_blob.lower()
    
    # Check for specific claims that should be evidence-based
    suspicious_patterns = [
        r"the user (definitely|certainly|clearly) (did|performed|executed)",
        r"this proves (that|the user)",
        r"(obviously|clearly|definitely) indicates",
        r"the evidence shows (beyond doubt|conclusively)",
        r"(criminal|malicious|illegal) (intent|activity|behavior)",
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, text_lower):
            # Check if there's supporting evidence
            key_terms = re.findall(r'\b\w+\b', pattern)
            if not any(term in evidence_lower for term in key_terms[-3:]):  # Check last few terms
                LOGGER.warning(f"Potentially unsupported claim detected: {pattern}")
                text += "\n\n[CAUTION: Some statements may require additional evidence verification]"
                break
    
    return text

def validate_llm_response(response: str, context: str, question: str) -> Dict[str, Any]:
    """Comprehensive LLM response validation"""
    
    validation_result = {
        "is_valid": True,
        "confidence": 1.0,
        "issues": [],
        "sanitized_response": response
    }
    
    if not response or len(response.strip()) < 10:
        validation_result["is_valid"] = False
        validation_result["issues"].append("Response too short or empty")
        return validation_result
    
    # Check for hallucination indicators
    if _GUARDRAIL_BANNED_RE.search(response):
        validation_result["issues"].append("Contains banned terms")
        validation_result["confidence"] *= 0.5
        validation_result["sanitized_response"] = sanitize_against_hallucinations(response, context)
    
    # Check for evidence grounding
    if "insufficient evidence" not in response.lower():
        # Response claims to have evidence - validate key claims
        response_lower = response.lower()
        context_lower = context.lower()
        
        # Extract potential factual claims (simplified)
        claims = re.findall(r'(user \w+|file \w+|process \w+|at \d{4}-\d{2}-\d{2})', response_lower)
        unsupported_claims = []
        
        for claim in claims:
            if claim not in context_lower:
                unsupported_claims.append(claim)
        
        if unsupported_claims:
            validation_result["issues"].append(f"Potentially unsupported claims: {unsupported_claims}")
            validation_result["confidence"] *= 0.7
    
    # Check response relevance to question
    question_terms = set(re.findall(r'\b\w+\b', question.lower()))
    response_terms = set(re.findall(r'\b\w+\b', response.lower()))
    
    relevance_score = len(question_terms & response_terms) / len(question_terms) if question_terms else 0
    if relevance_score < 0.3:
        validation_result["issues"].append("Low relevance to question")
        validation_result["confidence"] *= 0.8
    
    validation_result["confidence"] = max(0.1, validation_result["confidence"])
    
    return validation_result

# =============================================================================
# CSV INGESTION - Enhanced parallel processing
# =============================================================================

@performance_monitor
def process_single_csv(case_id: str, csv_path: Path, con: sqlite3.Connection) -> int:
    """Enhanced CSV processing with better error handling and performance"""
    
    if not csv_path.exists():
        LOGGER.error(f"CSV file not found: {csv_path}")
        return 0
    
    try:
        # Calculate file hash
        src_hash = sha256_file(csv_path)
        file_size = csv_path.stat().st_size
        
        # Check if already processed
        cur = con.execute("SELECT status FROM sources WHERE src_file = ? AND src_sha256 = ?", 
                         (str(csv_path), src_hash))
        existing = cur.fetchone()
        if existing and existing[0] == 'processed':
            LOGGER.info(f"Skipping already processed file: {csv_path}")
            return 0
        
    except Exception as e:
        LOGGER.error(f"Failed to hash file {csv_path}: {e}")
        src_hash = None
        file_size = 0
    
    # Update sources table
    con.execute(
        "INSERT OR REPLACE INTO sources(src_file, tool, tool_version, src_sha256, file_size, ingested_utc, status) VALUES (?,?,?,?,?,?,?)",
        (str(csv_path), "KAPE", "unknown", src_hash, file_size, int(time.time()), "processing")
    )
    
    artifact = detect_artifact(csv_path.name)
    LOGGER.info(f"Processing {csv_path} as {artifact}")
    
    try:
        # Enhanced CSV reading with better encoding detection
        encodings_to_try = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']
        df_reader = None
        
        for encoding in encodings_to_try:
            try:
                df_reader = pd.read_csv(
                    csv_path, 
                    header=0, 
                    low_memory=False, 
                    encoding=encoding,
                    chunksize=CONFIG.chunk_size,
                    na_values=['', 'NULL', 'null', 'N/A', 'n/a'],
                    keep_default_na=True
                )
                LOGGER.debug(f"Successfully opened {csv_path} with encoding {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                LOGGER.warning(f"Failed to read {csv_path} with encoding {encoding}: {e}")
                continue
        
        if df_reader is None:
            LOGGER.error(f"Failed to read CSV file with any encoding: {csv_path}")
            return 0
        
        # Ensure df_reader is iterable
        if not hasattr(df_reader, "__iter__"):
            df_reader = [df_reader]
        
    except Exception as e:
        LOGGER.error(f"Failed to open CSV file {csv_path}: {e}")
        return 0
    
    # Process chunks
    cur = con.cursor()
    row_counter = 0
    batch_data = []
    total_rows = 0
    
    try:
        for chunk_num, df in enumerate(df_reader):
            if df is None or df.empty:
                continue
            
            LOGGER.debug(f"Processing chunk {chunk_num + 1} with {len(df)} rows")
            
            # Check memory usage
            memory_info = check_memory_usage()
            if memory_info["percent"] > CONFIG.max_memory_usage:
                LOGGER.warning(f"High memory usage: {memory_info['percent']:.1%}, processing smaller batches")
                # Process current batch immediately
                if batch_data:
                    _execute_batch_insert(cur, batch_data)
                    batch_data = []
            
            records = df.to_dict(orient="records")
            
            for row in records:
                row_counter += 1
                
                # Enhanced timestamp extraction
                ts = pick_timestamp_optimized(row, str(csv_path), con)
                
                # Enhanced field extraction
                host = extract_host_info(row)
                user = extract_user_info(row)
                
                # Generate unique row ID
                row_id = sha256_text(f"{csv_path}:{row_counter}:{hash(str(sorted(row.items())))}")
                
                # Limit JSON field size to prevent database bloat
                fields_json = json.dumps(row, default=str, ensure_ascii=False)
                if len(fields_json) > 100000:  # 100KB limit
                    # Truncate large fields
                    truncated_row = {}
                    for k, v in row.items():
                        str_v = str(v)
                        if len(str_v) > 1000:
                            truncated_row[k] = str_v[:1000] + "...[TRUNCATED]"
                        else:
                            truncated_row[k] = v
                    fields_json = json.dumps(truncated_row, default=str, ensure_ascii=False)
                
                # Enhanced summary generation
                summary = build_summary(artifact, row)
                
                # Generate row hash for deduplication
                row_hash = sha256_text(f"{case_id}|{host}|{user}|{ts}|{artifact}|{summary}|{fields_json}|{src_hash}")
                
                batch_data.append((
                    row_id, case_id, host, user, ts, artifact, str(csv_path), 
                    summary, fields_json, src_hash, row_hash
                ))
                
                # Execute batch when it reaches the configured size
                if len(batch_data) >= CONFIG.batch_size:
                    _execute_batch_insert(cur, batch_data)
                    total_rows += len(batch_data)
                    batch_data = []
        
        # Execute remaining batch
        if batch_data:
            _execute_batch_insert(cur, batch_data)
            total_rows += len(batch_data)
        
        # Update source status
        con.execute(
            "UPDATE sources SET status = 'processed' WHERE src_file = ?",
            (str(csv_path),)
        )
        
        con.commit()
        LOGGER.info(f"Successfully processed {csv_path}: {total_rows} rows")
        return total_rows
        
    except Exception as e:
        LOGGER.error(f"Error processing CSV {csv_path}: {e}")
        con.rollback()
        # Update source status to failed
        con.execute(
            "UPDATE sources SET status = 'failed' WHERE src_file = ?",
            (str(csv_path),)
        )
        con.commit()
        return 0

def _execute_batch_insert(cur: sqlite3.Cursor, batch_data: List[Tuple]):
    """Execute batch insert with error handling"""
    try:
        cur.executemany(
            """
            INSERT OR REPLACE INTO evidence(
                row_id, case_id, host, user, ts_utc, artifact, src_file, 
                summary, fields_json, src_sha256, row_sha256
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            batch_data
        )
    except Exception as e:
        LOGGER.error(f"Batch insert failed: {e}")
        # Try individual inserts to identify problematic rows
        for i, row_data in enumerate(batch_data):
            try:
                cur.execute(
                    """
                    INSERT OR REPLACE INTO evidence(
                        row_id, case_id, host, user, ts_utc, artifact, src_file, 
                        summary, fields_json, src_sha256, row_sha256
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    row_data
                )
            except Exception as row_e:
                LOGGER.error(f"Failed to insert row {i}: {row_e}")

def extract_host_info(row: Dict[str, Any]) -> Optional[str]:
    """Enhanced host information extraction"""
    host_fields = [
        "Computer", "Host", "ComputerName", "Hostname", "MachineName", 
        "SystemName", "NodeName", "ServerName"
    ]
    
    for field in host_fields:
        if field in row and row[field]:
            return str(row[field]).strip()
    
    return None

def extract_user_info(row: Dict[str, Any]) -> Optional[str]:
    """Enhanced user information extraction"""
    user_fields = [
        "User", "Username", "UserName", "AccountName", "Account",
        "SubjectUserName", "TargetUserName", "SamAccountName", 
        "LogonUser", "Owner", "CreatedBy", "ModifiedBy"
    ]
    
    for field in user_fields:
        if field in row and row[field]:
            user = str(row[field]).strip()
            # Filter out system accounts and empty values
            if user and user.lower() not in ['system', 'null', 'n/a', '', '-']:
                return user
    
    return None

@performance_monitor
def ingest_extracts_parallel(con: sqlite3.Connection, case_id: str, 
                           extracts_dir: Path = None) -> int:
    """Enhanced parallel CSV ingestion with better resource management"""
    
    if extracts_dir is None:
        extracts_dir = CONFIG.dir_extracts
    
    if not extracts_dir.exists():
        LOGGER.error(f"Extracts directory not found: {extracts_dir}")
        return 0
    
    # Initialize database
    initialize_database(con)
    
    # Find all CSV files
    csv_files = list(extracts_dir.rglob("*.csv"))
    if not csv_files:
        LOGGER.warning(f"No CSV files found in {extracts_dir}")
        return 0
    
    LOGGER.info(f"Found {len(csv_files)} CSV files to process")
    
    # Sort by file size (process smaller files first for better progress indication)
    csv_files.sort(key=lambda p: p.stat().st_size)
    
    total_rows = 0
    processed_files = 0
    
    # Use ThreadPoolExecutor for I/O bound CSV processing
    with ThreadPoolExecutor(max_workers=CONFIG.max_workers) as executor:
        # Submit all tasks
        future_to_file = {}
        for csv_file in csv_files:
            # Create a new connection for each thread
            thread_con = db_connect()
            future = executor.submit(process_single_csv, case_id, csv_file, thread_con)
            future_to_file[future] = (csv_file, thread_con)
        
        # Process completed tasks
        if HAVE_TQDM:
            progress_bar = tqdm(total=len(csv_files), desc="Processing CSV files")
        
        for future in as_completed(future_to_file):
            csv_file, thread_con = future_to_file[future]
            try:
                rows_processed = future.result()
                total_rows += rows_processed
                processed_files += 1
                
                if HAVE_TQDM:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'files': f"{processed_files}/{len(csv_files)}",
                        'rows': total_rows
                    })
                else:
                    LOGGER.info(f"Progress: {processed_files}/{len(csv_files)} files, {total_rows} total rows")
                
            except Exception as e:
                LOGGER.error(f"Failed to process {csv_file}: {e}")
            finally:
                thread_con.close()
        
        if HAVE_TQDM:
            progress_bar.close()
    
    LOGGER.info(f"Parallel ingestion complete: {processed_files} files, {total_rows} total rows")
    return total_rows

# =============================================================================
# KAPE INTEGRATION - Enhanced external tool orchestration
# =============================================================================

def check_kape_prereqs():
    """Enhanced KAPE prerequisites check"""
    issues = []
    
    if not CONFIG.kape_exe.exists():
        issues.append(f"KAPE executable not found: {CONFIG.kape_exe}")
    
    if not CONFIG.sqle_maps.exists():
        issues.append(f"SQLECmd maps not found: {CONFIG.sqle_maps}")
    
    # Check for required directories
    required_dirs = [CONFIG.dir_artifacts, CONFIG.dir_extracts]
    for directory in required_dirs:
        if not directory.exists():
            try:
                directory.mkdir(parents=True, exist_ok=True)
                LOGGER.info(f"Created directory: {directory}")
            except Exception as e:
                issues.append(f"Cannot create directory {directory}: {e}")
    
    if issues:
        error_msg = "KAPE prerequisites check failed:\n" + "\n".join(f"- {issue}" for issue in issues)
        LOGGER.error(error_msg)
        raise RuntimeError(error_msg)
    
    LOGGER.info("KAPE prerequisites check passed")

@performance_monitor
def kape_collect(tsource: str):
    """Enhanced KAPE collection with better error handling"""
    
    cmd = [
        str(CONFIG.kape_exe),
        "--tsource", tsource,
        "--tdest", str(CONFIG.dir_artifacts),
        "--target", "!SANS_Triage",
        "--vhdx", str(CONFIG.dir_artifacts / "collection.vhdx"),
        "--zip", str(CONFIG.dir_artifacts / "collection.zip")
    ]
    
    try:
        result = run_command(cmd, timeout=3600)  # 1 hour timeout
        
        if result.returncode != 0:
            LOGGER.error(f"KAPE collection failed: {result.stderr}")
            raise RuntimeError(f"KAPE collection failed with code {result.returncode}")
        
        LOGGER.info("KAPE collection completed successfully")
        
    except subprocess.TimeoutExpired:
        LOGGER.error("KAPE collection timed out")
        raise
    except Exception as e:
        LOGGER.error(f"KAPE collection error: {e}")
        raise

@performance_monitor
def kape_parse():
    """Enhanced KAPE parsing with better error handling"""
    
    cmd = [
        str(CONFIG.kape_exe),
        "--msource", str(CONFIG.dir_artifacts),
        "--mdest", str(CONFIG.dir_extracts),
        "--module", "!EZParser"
    ]
    
    try:
        result = run_command(cmd, timeout=7200)  # 2 hour timeout
        
        if result.returncode != 0:
            LOGGER.error(f"KAPE parsing failed: {result.stderr}")
            raise RuntimeError(f"KAPE parsing failed with code {result.returncode}")
        
        LOGGER.info("KAPE parsing completed successfully")
        
    except subprocess.TimeoutExpired:
        LOGGER.error("KAPE parsing timed out")
        raise
    except Exception as e:
        LOGGER.error(f"KAPE parsing error: {e}")
        raise

def run_live_only_supplements():
    """Enhanced live system supplemental commands"""
    
    supplements_dir = CONFIG.dir_extracts / "Supplements"
    supplements_dir.mkdir(exist_ok=True)
    
    # Enhanced command list with better error handling
    commands = [
        {
            "name": "systeminfo",
            "cmd": ["systeminfo"],
            "output": supplements_dir / "systeminfo.txt"
        },
        {
            "name": "tasklist",
            "cmd": ["tasklist", "/v"],
            "output": supplements_dir / "tasklist.txt"
        },
        {
            "name": "netstat",
            "cmd": ["netstat", "-ano"],
            "output": supplements_dir / "netstat.txt"
        },
        {
            "name": "ipconfig",
            "cmd": ["ipconfig", "/all"],
            "output": supplements_dir / "ipconfig.txt"
        },
        {
            "name": "wmic_process",
            "cmd": ["wmic", "process", "list", "full"],
            "output": supplements_dir / "wmic_process.txt"
        },
        {
            "name": "wmic_service",
            "cmd": ["wmic", "service", "list", "full"],
            "output": supplements_dir / "wmic_service.txt"
        }
    ]
    
    for cmd_info in commands:
        try:
            LOGGER.info(f"Running {cmd_info['name']}...")
            result = run_command(cmd_info["cmd"], timeout=60)
            
            with cmd_info["output"].open('w', encoding='utf-8', errors='replace') as f:
                f.write(f"Command: {' '.join(cmd_info['cmd'])}\n")
                f.write(f"Exit Code: {result.returncode}\n")
                f.write(f"Timestamp: {datetime.now(timezone.utc).isoformat()}\n")
                f.write("-" * 50 + "\n")
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n" + "-" * 50 + "\n")
                    f.write("STDERR:\n")
                    f.write(result.stderr)
            
            LOGGER.info(f"Completed {cmd_info['name']}")
            
        except Exception as e:
            LOGGER.error(f"Failed to run {cmd_info['name']}: {e}")

def copy_setupapi_logs():
    """Enhanced setupapi log copying with better error handling"""
    
    registry_dir = CONFIG.dir_extracts / "Registry"
    registry_dir.mkdir(exist_ok=True)
    
    # Common setupapi.log locations
    setupapi_paths = [
        Path("C:/Windows/setupapi.log"),
        Path("C:/Windows/inf/setupapi.log"),
        Path("C:/Windows/System32/setupapi.log"),
        CONFIG.dir_artifacts / "setupapi.log"
    ]
    
    copied_count = 0
    
    for src_path in setupapi_paths:
        if src_path.exists():
            try:
                dest_path = registry_dir / f"setupapi_{src_path.parent.name}.log"
                shutil.copy2(src_path, dest_path)
                LOGGER.info(f"Copied {src_path} to {dest_path}")
                copied_count += 1
            except Exception as e:
                LOGGER.warning(f"Failed to copy {src_path}: {e}")
    
    if copied_count == 0:
        LOGGER.warning("No setupapi.log files found to copy")
    else:
        LOGGER.info(f"Copied {copied_count} setupapi.log files")

# =============================================================================
# FORENSIC ANALYSIS - Enhanced question answering logic
# =============================================================================

def get_header_values(con: sqlite3.Connection, range_text: str) -> Dict[str, str]:
    """Enhanced header information extraction"""
    
    header = {
        "CaseId": "",
        "ComputerName": "Unknown",
        "DateRange": range_text,
        "AnalysisTime": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        "TotalEvidence": "0",
        "ArtifactTypes": "0"
    }
    
    try:
        # Get case ID from evidence
        cur = con.execute("SELECT DISTINCT case_id FROM evidence LIMIT 1")
        row = cur.fetchone()
        if row:
            header["CaseId"] = row[0]
        
        # Get computer name from system info
        cur = con.execute("""
            SELECT computer_name FROM mv_computer_identity 
            WHERE computer_name IS NOT NULL 
            ORDER BY confidence_score DESC 
            LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            header["ComputerName"] = row[0]
        
        # Get evidence statistics
        cur = con.execute("SELECT COUNT(*) FROM evidence")
        row = cur.fetchone()
        if row:
            header["TotalEvidence"] = str(row[0])
        
        cur = con.execute("SELECT COUNT(DISTINCT artifact) FROM evidence")
        row = cur.fetchone()
        if row:
            header["ArtifactTypes"] = str(row[0])
        
    except Exception as e:
        LOGGER.error(f"Failed to get header values: {e}")
    
    return header

@performance_monitor
def answer_questions(con: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Enhanced forensic question answering with better accuracy"""
    
    qa_results = []
    
    for i, question in enumerate(FORENSIC_QUESTIONS, 1):
        LOGGER.info(f"Answering question {i}: {question}")
        
        try:
            answer = answer_single_question(con, question)
            qa_results.append({
                "question_num": i,
                "question": question,
                "answer": answer["text"],
                "confidence": answer.get("confidence", 0.5),
                "evidence_count": answer.get("evidence_count", 0),
                "sources": answer.get("sources", [])
            })
        except Exception as e:
            LOGGER.error(f"Failed to answer question {i}: {e}")
            qa_results.append({
                "question_num": i,
                "question": question,
                "answer": f"Error processing question: {e}",
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": []
            })
    
    return qa_results

def answer_single_question(con: sqlite3.Connection, question: str) -> Dict[str, Any]:
    """Answer a single forensic question with enhanced accuracy"""
    
    # Question-specific logic for better accuracy
    question_lower = question.lower()
    
    if "computername" in question_lower or "computer name" in question_lower:
        return answer_computer_name_question(con)
    elif "make" in question_lower and "model" in question_lower and "serial" in question_lower:
        return answer_system_info_question(con)
    elif "hard drive" in question_lower or "drive" in question_lower:
        return answer_drive_info_question(con)
    elif "username" in question_lower or "user" in question_lower and "sid" in question_lower:
        return answer_user_accounts_question(con)
    elif "primary user" in question_lower:
        return answer_primary_user_question(con)
    elif "tampering" in question_lower or "destruction" in question_lower:
        return answer_tampering_question(con)
    elif "removable" in question_lower or "usb" in question_lower:
        return answer_usb_devices_question(con)
    elif "files" in question_lower and ("copied" in question_lower or "transferred" in question_lower):
        return answer_file_transfer_question(con)
    elif "cloud" in question_lower:
        return answer_cloud_storage_question(con)
    elif "screenshot" in question_lower:
        return answer_screenshot_question(con)
    elif "print" in question_lower:
        return answer_printing_question(con)
    elif "software" in question_lower or "install" in question_lower:
        return answer_software_question(con)
    else:
        # Generic question answering using FTS
        return answer_generic_question(con, question)

def answer_computer_name_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer computer name question with enhanced accuracy"""
    
    try:
        cur = con.execute("""
            SELECT computer_name, confidence_score, COUNT(*) as evidence_count
            FROM mv_computer_identity 
            WHERE computer_name IS NOT NULL 
            GROUP BY computer_name
            ORDER BY confidence_score DESC, evidence_count DESC
            LIMIT 5
        """)
        
        results = cur.fetchall()
        
        if not results:
            return {
                "text": "Insufficient evidence in scope.",
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": []
            }
        
        # Build answer with confidence levels
        if len(results) == 1:
            computer_name = results[0][0]
            evidence_count = results[0][2]
            answer = f"Computer name: {computer_name} (based on {evidence_count} evidence entries)"
            confidence = 0.9
        else:
            # Multiple computer names found
            primary = results[0]
            others = results[1:]
            answer = f"Primary computer name: {primary[0]} (confidence: high, {primary[2]} evidence entries)"
            if others:
                answer += f"\nOther names found: {', '.join([r[0] for r in others])}"
            confidence = 0.7
        
        return {
            "text": answer,
            "confidence": confidence,
            "evidence_count": sum(r[2] for r in results),
            "sources": ["mv_computer_identity"]
        }
        
    except Exception as e:
        LOGGER.error(f"Error answering computer name question: {e}")
        return {
            "text": f"Error processing question: {e}",
            "confidence": 0.0,
            "evidence_count": 0,
            "sources": []
        }

def answer_system_info_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer system information question"""
    
    try:
        cur = con.execute("""
            SELECT make, model, serial, COUNT(*) as evidence_count
            FROM mv_computer_identity 
            WHERE make IS NOT NULL OR model IS NOT NULL OR serial IS NOT NULL
            GROUP BY make, model, serial
            ORDER BY evidence_count DESC
            LIMIT 1
        """)
        
        result = cur.fetchone()
        
        if not result:
            return {
                "text": "Insufficient evidence in scope.",
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": []
            }
        
        make, model, serial, evidence_count = result
        
        parts = []
        if make:
            parts.append(f"Make: {make}")
        if model:
            parts.append(f"Model: {model}")
        if serial:
            parts.append(f"Serial: {serial}")
        
        if not parts:
            return {
                "text": "Insufficient evidence in scope.",
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": []
            }
        
        answer = "Computer system information:\n" + "\n".join(f"- {part}" for part in parts)
        answer += f"\n(Based on {evidence_count} evidence entries)"
        
        confidence = 0.8 if len(parts) >= 2 else 0.6
        
        return {
            "text": answer,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "sources": ["mv_computer_identity"]
        }
        
    except Exception as e:
        LOGGER.error(f"Error answering system info question: {e}")
        return {
            "text": f"Error processing question: {e}",
            "confidence": 0.0,
            "evidence_count": 0,
            "sources": []
        }

def answer_drive_info_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer drive information question"""
    
    try:
        cur = con.execute("""
            SELECT drive_model, drive_serial, COUNT(*) as evidence_count
            FROM mv_computer_identity 
            WHERE drive_model IS NOT NULL OR drive_serial IS NOT NULL
            GROUP BY drive_model, drive_serial
            ORDER BY evidence_count DESC
            LIMIT 5
        """)
        
        results = cur.fetchall()
        
        if not results:
            return {
                "text": "Insufficient evidence in scope.",
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": []
            }
        
        drives = []
        total_evidence = 0
        
        for drive_model, drive_serial, evidence_count in results:
            drive_info = []
            if drive_model:
                drive_info.append(f"Model: {drive_model}")
            if drive_serial:
                drive_info.append(f"Serial: {drive_serial}")
            
            if drive_info:
                drives.append(" | ".join(drive_info))
                total_evidence += evidence_count
        
        if not drives:
            return {
                "text": "Insufficient evidence in scope.",
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": []
            }
        
        answer = "Internal hard drive information:\n"
        for i, drive in enumerate(drives, 1):
            answer += f"- Drive {i}: {drive}\n"
        
        answer += f"(Based on {total_evidence} evidence entries)"
        
        confidence = 0.8 if len(drives) >= 1 else 0.6
        
        return {
            "text": answer,
            "confidence": confidence,
            "evidence_count": total_evidence,
            "sources": ["mv_computer_identity"]
        }
        
    except Exception as e:
        LOGGER.error(f"Error answering drive info question: {e}")
        return {
            "text": f"Error processing question: {e}",
            "confidence": 0.0,
            "evidence_count": 0,
            "sources": []
        }

def answer_user_accounts_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer user accounts question with enhanced detail"""
    
    try:
        cur = con.execute("""
            SELECT user, sid, account_type, first_activity, last_activity, evidence_count
            FROM mv_accounts_activity 
            WHERE user IS NOT NULL
            ORDER BY last_activity DESC, evidence_count DESC
            LIMIT 20
        """)
        
        results = cur.fetchall()
        
        if not results:
            return {
                "text": "Insufficient evidence in scope.",
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": []
            }
        
        # Group by account type
        account_groups = defaultdict(list)
        total_evidence = 0
        
        for user, sid, account_type, first_activity, last_activity, evidence_count in results:
            first_dt = datetime.fromtimestamp(first_activity, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if first_activity else "Unknown"
            last_dt = datetime.fromtimestamp(last_activity, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if last_activity else "Unknown"
            
            account_groups[account_type].append({
                "user": user,
                "sid": sid,
                "first": first_dt,
                "last": last_dt,
                "count": evidence_count
            })
            total_evidence += evidence_count
        
        # Build answer
        answer_parts = ["User accounts found:"]
        
        # Order account types by importance
        type_order = ["domain_user", "builtin", "system", "local_service", "network_service", "unknown"]
        
        for account_type in type_order:
            if account_type in account_groups:
                accounts = account_groups[account_type]
                answer_parts.append(f"\n{account_type.replace('_', ' ').title()} Accounts:")
                
                for account in accounts[:10]:  # Limit to top 10 per type
                    answer_parts.append(f"- {account['user']}")
                    if account['sid']:
                        answer_parts.append(f"  SID: {account['sid']}")
                    answer_parts.append(f"  First activity: {account['first']}")
                    answer_parts.append(f"  Last activity: {account['last']}")
                    answer_parts.append(f"  Evidence entries: {account['count']}")
        
        answer_parts.append(f"\nTotal evidence entries: {total_evidence}")
        
        confidence = 0.8 if len(results) >= 3 else 0.6
        
        return {
            "text": "\n".join(answer_parts),
            "confidence": confidence,
            "evidence_count": total_evidence,
            "sources": ["mv_accounts_activity"]
        }
        
    except Exception as e:
        LOGGER.error(f"Error answering user accounts question: {e}")
        return {
            "text": f"Error processing question: {e}",
            "confidence": 0.0,
            "evidence_count": 0,
            "sources": []
        }

def answer_primary_user_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer primary user question"""
    
    try:
        cur = con.execute("""
            SELECT user, sid, first_activity, last_activity, evidence_count, artifact_types
            FROM mv_primary_user
            LIMIT 1
        """)
        
        result = cur.fetchone()
        
        if not result:
            return {
                "text": "Insufficient evidence in scope to determine primary user.",
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": []
            }
        
        user, sid, first_activity, last_activity, evidence_count, artifact_types = result
        
        first_dt = datetime.fromtimestamp(first_activity, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if first_activity else "Unknown"
        last_dt = datetime.fromtimestamp(last_activity, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if last_activity else "Unknown"
        
        answer = f"Primary user: {user}\n"
        if sid:
            answer += f"SID: {sid}\n"
        answer += f"First activity: {first_dt}\n"
        answer += f"Last activity: {last_dt}\n"
        answer += f"Evidence entries: {evidence_count}\n"
        answer += f"Artifact types: {artifact_types}"
        
        # Confidence based on evidence volume and recency
        confidence = min(0.9, 0.5 + (evidence_count / 1000) + (artifact_types / 10))
        
        return {
            "text": answer,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "sources": ["mv_primary_user"]
        }
        
    except Exception as e:
        LOGGER.error(f"Error answering primary user question: {e}")
        return {
            "text": f"Error processing question: {e}",
            "confidence": 0.0,
            "evidence_count": 0,
            "sources": []
        }

def answer_tampering_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer tampering/destruction question"""
    
    try:
        cur = con.execute("""
            SELECT ts_utc, user, tamper_type, summary, src_file
            FROM mv_tamper_evidence
            ORDER BY ts_utc DESC
            LIMIT 20
        """)
        
        results = cur.fetchall()
        
        if not results:
            return {
                "text": "No evidence of data destruction or forensic tampering found in scope.",
                "confidence": 0.7,
                "evidence_count": 0,
                "sources": ["mv_tamper_evidence"]
            }
        
        # Group by tamper type
        tamper_groups = defaultdict(list)
        
        for ts_utc, user, tamper_type, summary, src_file in results:
            timestamp = datetime.fromtimestamp(ts_utc, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S') if ts_utc else "Unknown"
            tamper_groups[tamper_type].append({
                "timestamp": timestamp,
                "user": user or "Unknown",
                "summary": summary,
                "source": Path(src_file).name if src_file else "Unknown"
            })
        
        answer_parts = ["Evidence of potential tampering found:"]
        
        for tamper_type, events in tamper_groups.items():
            answer_parts.append(f"\n{tamper_type.replace('_', ' ').title()}:")
            for event in events[:5]:  # Limit to top 5 per type
                answer_parts.append(f"- {event['timestamp']} | User: {event['user']}")
                answer_parts.append(f"  Activity: {event['summary']}")
                answer_parts.append(f"  Source: {event['source']}")
        
        answer_parts.append(f"\nTotal suspicious activities: {len(results)}")
        
        confidence = 0.8 if len(results) >= 3 else 0.6
        
        return {
            "text": "\n".join(answer_parts),
            "confidence": confidence,
            "evidence_count": len(results),
            "sources": ["mv_tamper_evidence"]
        }
        
    except Exception as e:
        LOGGER.error(f"Error answering tampering question: {e}")
        return {
            "text": f"Error processing question: {e}",
            "confidence": 0.0,
            "evidence_count": 0,
            "sources": []
        }

def answer_usb_devices_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer USB devices question"""
    
    try:
        cur = con.execute("""
            SELECT ts_utc, user, make, model, serial, drive_letter, artifact
            FROM mv_usb_devices
            WHERE make IS NOT NULL OR model IS NOT NULL OR serial IS NOT NULL
            ORDER BY ts_utc DESC
            LIMIT 20
        """)
        
        results = cur.fetchall()
        
        if not results:
            return {
                "text": "No removable storage devices found in scope.",
                "confidence": 0.7,
                "evidence_count": 0,
                "sources": ["mv_usb_devices"]
            }
        
        # Group by device (make/model/serial combination)
        devices = {}
        
        for ts_utc, user, make, model, serial, drive_letter, artifact in results:
            device_key = f"{make or 'Unknown'}|{model or 'Unknown'}|{serial or 'Unknown'}"
            
            if device_key not in devices:
                devices[device_key] = {
                    "make": make or "Unknown",
                    "model": model or "Unknown", 
                    "serial": serial or "Unknown",
                    "first_seen": ts_utc,
                    "last_seen": ts_utc,
                    "users": set(),
                    "drive_letters": set(),
                    "artifacts": set()
                }
            
            device = devices[device_key]
            if ts_utc:
                device["first_seen"] = min(device["first_seen"] or float('inf'), ts_utc)
                device["last_seen"] = max(device["last_seen"] or 0, ts_utc)
            
            if user:
                device["users"].add(user)
            if drive_letter:
                device["drive_letters"].add(drive_letter)
            if artifact:
                device["artifacts"].add(artifact)
        
        answer_parts = ["Removable storage devices found:"]
        
        for i, (device_key, device) in enumerate(devices.items(), 1):
            answer_parts.append(f"\nDevice {i}:")
            answer_parts.append(f"- Make: {device['make']}")
            answer_parts.append(f"- Model: {device['model']}")
            answer_parts.append(f"- Serial: {device['serial']}")
            
            if device['first_seen']:
                first_dt = datetime.fromtimestamp(device['first_seen'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                answer_parts.append(f"- First seen: {first_dt}")
            
            if device['last_seen']:
                last_dt = datetime.fromtimestamp(device['last_seen'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                answer_parts.append(f"- Last seen: {last_dt}")
            
            if device['users']:
                answer_parts.append(f"- Users: {', '.join(sorted(device['users']))}")
            
            if device['drive_letters']:
                answer_parts.append(f"- Drive letters: {', '.join(sorted(device['drive_letters']))}")
        
        answer_parts.append(f"\nTotal devices: {len(devices)}")
        
        confidence = 0.8 if len(devices) >= 1 else 0.6
        
        return {
            "text": "\n".join(answer_parts),
            "confidence": confidence,
            "evidence_count": len(results),
            "sources": ["mv_usb_devices"]
        }
        
    except Exception as e:
        LOGGER.error(f"Error answering USB devices question: {e}")
        return {
            "text": f"Error processing question: {e}",
            "confidence": 0.0,
            "evidence_count": 0,
            "sources": []
        }

def answer_generic_question(con: sqlite3.Connection, question: str) -> Dict[str, Any]:
    """Generic question answering using FTS search"""
    
    try:
        # Use enhanced FTS search
        evidence_rows = enhanced_fts_search(con, question, limit=100)
        
        if not evidence_rows:
            return {
                "text": "Insufficient evidence in scope.",
                "confidence": 0.0,
                "evidence_count": 0,
                "sources": []
            }
        
        # Build context and analyze
        context = build_comprehensive_context(evidence_rows, max_context_size=4000)
        
        # Simple pattern-based analysis for common question types
        answer_parts = []
        
        # Extract key information from evidence
        timestamps = []
        users = set()
        files = set()
        processes = set()
        
        for row in evidence_rows[:20]:  # Analyze top 20 results
            if row.get('ts_utc'):
                timestamps.append(row['ts_utc'])
            if row.get('user'):
                users.add(row['user'])
            
            # Extract additional info from JSON fields
            if row.get('fields_json'):
                try:
                    fields = json.loads(row['fields_json'])
                    for field in ['FileName', 'Path', 'ProcessName']:
                        if field in fields and fields[field]:
                            if field in ['FileName', 'Path']:
                                files.add(str(fields[field]))
                            elif field == 'ProcessName':
                                processes.add(str(fields[field]))
                except:
                    pass
        
        # Build summary answer
        answer_parts.append(f"Found {len(evidence_rows)} relevant evidence entries.")
        
        if timestamps:
            earliest = min(timestamps)
            latest = max(timestamps)
            earliest_dt = datetime.fromtimestamp(earliest, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            latest_dt = datetime.fromtimestamp(latest, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            answer_parts.append(f"Time range: {earliest_dt} to {latest_dt}")
        
        if users:
            answer_parts.append(f"Users involved: {', '.join(sorted(list(users)[:5]))}")
        
        if files:
            answer_parts.append(f"Files mentioned: {', '.join(sorted(list(files)[:5]))}")
        
        if processes:
            answer_parts.append(f"Processes mentioned: {', '.join(sorted(list(processes)[:5]))}")
        
        # Add top evidence summaries
        answer_parts.append("\nTop evidence:")
        for i, row in enumerate(evidence_rows[:5], 1):
            timestamp = ""
            if row.get('ts_utc'):
                dt = datetime.fromtimestamp(row['ts_utc'], tz=timezone.utc)
                timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
            
            summary = row.get('summary', 'No summary')
            answer_parts.append(f"{i}. {timestamp} - {summary}")
        
        confidence = min(0.8, 0.3 + (len(evidence_rows) / 100))
        
        return {
            "text": "\n".join(answer_parts),
            "confidence": confidence,
            "evidence_count": len(evidence_rows),
            "sources": ["enhanced_fts_search"]
        }
        
    except Exception as e:
        LOGGER.error(f"Error answering generic question: {e}")
        return {
            "text": f"Error processing question: {e}",
            "confidence": 0.0,
            "evidence_count": 0,
            "sources": []
        }

# Placeholder implementations for remaining question types
def answer_file_transfer_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer file transfer question - placeholder implementation"""
    return answer_generic_question(con, "file transfer copy usb removable")

def answer_cloud_storage_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer cloud storage question - placeholder implementation"""
    return answer_generic_question(con, "cloud storage dropbox onedrive google drive")

def answer_screenshot_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer screenshot question - placeholder implementation"""
    return answer_generic_question(con, "screenshot screen capture snipping tool")

def answer_printing_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer printing question - placeholder implementation"""
    return answer_generic_question(con, "print printer spool document")

def answer_software_question(con: sqlite3.Connection) -> Dict[str, Any]:
    """Answer software installation question - placeholder implementation"""
    return answer_generic_question(con, "install software program service")

# =============================================================================
# LLM FUNCTIONS - Enhanced language model integration
# =============================================================================

def build_enhanced_llm_prompt(header: Dict[str, str], qa: List[Dict[str, Any]]) -> str:
    """Build enhanced LLM prompt with better structure and context"""
    
    lines = []
    lines.append("FORENSIC ANALYSIS EXECUTIVE SUMMARY REQUEST")
    lines.append("=" * 60)
    lines.append("")
    
    # Case context
    lines.append("CASE INFORMATION:")
    lines.append(f"Case ID: {header.get('CaseId', 'Unknown')}")
    lines.append(f"Computer: {header.get('ComputerName', 'Unknown')}")
    lines.append(f"Analysis Period: {header.get('DateRange', 'Unknown')}")
    lines.append(f"Total Evidence: {header.get('TotalEvidence', '0')} entries")
    lines.append(f"Artifact Types: {header.get('ArtifactTypes', '0')}")
    lines.append(f"Analysis Time: {header.get('AnalysisTime', 'Unknown')}")
    lines.append("")
    
    # Question and answer pairs
    lines.append("FORENSIC ANALYSIS RESULTS:")
    lines.append("-" * 40)
    
    for qa_item in qa:
        lines.append(f"\nQ{qa_item['question_num']}: {qa_item['question']}")
        lines.append(f"A{qa_item['question_num']}: {qa_item['answer']}")
        
        # Add confidence and evidence metadata
        confidence = qa_item.get('confidence', 0.0)
        evidence_count = qa_item.get('evidence_count', 0)
        lines.append(f"[Confidence: {confidence:.1%}, Evidence: {evidence_count} entries]")
    
    lines.append("")
    lines.append("EXECUTIVE SUMMARY REQUIREMENTS:")
    lines.append("1. Provide a concise executive summary of the forensic findings")
    lines.append("2. Highlight the most significant discoveries and their implications")
    lines.append("3. Organize findings by importance and relevance")
    lines.append("4. Use only the evidence provided above - do not speculate")
    lines.append("5. Note any limitations or gaps in the evidence")
    lines.append("6. Maintain professional, neutral language throughout")
    lines.append("7. Structure the summary for executive-level consumption")
    
    return "\n".join(lines)

@performance_monitor
def llm_summarize_enhanced(header: Dict[str, str], qa: List[Dict[str, Any]], 
                          model_path: str, max_tokens: int = 1200) -> str:
    """Enhanced LLM summarization with better accuracy and error handling"""
    
    if not HAVE_LLAMA:
        error_msg = "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        LOGGER.error(error_msg)
        raise RuntimeError(error_msg)
    
    if not Path(model_path).exists():
        error_msg = f"LLM model not found: {model_path}"
        LOGGER.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Initialize LLM with enhanced configuration
        llm = Llama(
            model_path=model_path,
            n_ctx=CONFIG.llm_context_size,
            chat_format="chatml",
            n_threads=CONFIG.llm_threads,
            verbose=False,
            n_batch=512,  # Batch size for processing
            use_mmap=True,  # Memory mapping for efficiency
            use_mlock=False,  # Don't lock memory
        )
        
        prompt = build_enhanced_llm_prompt(header, qa)
        
        # Enhanced generation parameters
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=CONFIG.llm_temperature,
            top_p=CONFIG.llm_top_p,
            repeat_penalty=1.1,
            stop=["</s>", "Human:", "Assistant:", "User:"],
            stream=False
        )
        
        text = response["choices"][0]["message"]["content"].strip()
        
        # Enhanced validation and sanitization
        validation_result = validate_llm_response(text, prompt, "executive summary")
        
        if not validation_result["is_valid"]:
            LOGGER.warning(f"LLM response validation issues: {validation_result['issues']}")
        
        sanitized_text = validation_result["sanitized_response"]
        
        # Add metadata footer
        sanitized_text += f"\n\n[Generated by {Path(model_path).name}]"
        sanitized_text += f"\n[Confidence: {validation_result['confidence']:.1%}]"
        if validation_result["issues"]:
            sanitized_text += f"\n[Validation notes: {'; '.join(validation_result['issues'])}]"
        
        return sanitized_text
        
    except Exception as e:
        LOGGER.error(f"LLM summarization failed: {e}")
        raise

@performance_monitor
def llm_answer_custom_enhanced(con: sqlite3.Connection, header: Dict[str, str], 
                              nl_question: str, model_path: str, max_tokens: int = 1000) -> str:
    """Enhanced ad-hoc question answering with comprehensive context"""
    
    if not HAVE_LLAMA:
        error_msg = "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        LOGGER.error(error_msg)
        raise RuntimeError(error_msg)
    
    if not Path(model_path).exists():
        error_msg = f"LLM model not found: {model_path}"
        LOGGER.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # Enhanced context gathering with multiple search strategies
        context_hits = enhanced_fts_search(con, nl_question, limit=200)
        
        # Build comprehensive context
        context_text = build_comprehensive_context(context_hits, max_context_size=6000)
        
        # Build enhanced prompt
        lines = []
        lines.append(f"FORENSIC ANALYSIS QUESTION: {nl_question}")
        lines.append("")
        lines.append("CASE CONTEXT:")
        lines.append(f"Computer: {header.get('ComputerName', 'Unknown')}")
        lines.append(f"Analysis Period: {header.get('DateRange', 'Unknown')}")
        lines.append(f"Total Evidence: {header.get('TotalEvidence', '0')} entries")
        lines.append("")
        lines.append("RELEVANT EVIDENCE:")
        lines.append(context_text)
        lines.append("")
        lines.append("ANALYSIS REQUIREMENTS:")
        lines.append("1. Answer the question using ONLY the evidence provided above")
        lines.append("2. If evidence is insufficient, state exactly: 'Insufficient evidence in scope'")
        lines.append("3. Provide specific timestamps, file names, and user accounts when available")
        lines.append("4. Do NOT speculate beyond the evidence or assert criminal intent")
        lines.append("5. Distinguish between correlation and causation")
        lines.append("6. Note any limitations in the available evidence")
        lines.append("7. Structure your response clearly with bullet points when appropriate")
        
        prompt = "\n".join(lines)
        
        # Initialize LLM
        llm = Llama(
            model_path=model_path,
            n_ctx=CONFIG.llm_context_size,
            chat_format="chatml",
            n_threads=CONFIG.llm_threads,
            verbose=False,
            use_mmap=True,
        )
        
        # Generate response
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": GUARDRAIL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=CONFIG.llm_temperature,
            top_p=CONFIG.llm_top_p,
            repeat_penalty=1.1,
            stop=["</s>", "Human:", "Assistant:", "User:"],
            stream=False
        )
        
        text = response["choices"][0]["message"]["content"].strip()
        
        # Enhanced validation
        validation_result = validate_llm_response(text, context_text, nl_question)
        
        if not validation_result["is_valid"]:
            LOGGER.warning(f"LLM response validation issues: {validation_result['issues']}")
        
        sanitized_text = validation_result["sanitized_response"]
        
        # Add metadata
        sanitized_text += f"\n\n[Evidence entries analyzed: {len(context_hits)}]"
        sanitized_text += f"\n[Response confidence: {validation_result['confidence']:.1%}]"
        if validation_result["issues"]:
            sanitized_text += f"\n[Validation notes: {'; '.join(validation_result['issues'])}]"
        
        return sanitized_text
        
    except Exception as e:
        LOGGER.error(f"LLM custom question answering failed: {e}")
        raise

# =============================================================================
# OUTPUT AND REPORTING - Enhanced report generation
# =============================================================================

@performance_monitor
def write_outputs(case_id: str, header: Dict[str, str], qa: List[Dict[str, Any]]) -> Dict[str, str]:
    """Enhanced output generation with multiple formats"""
    
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    outputs = {}
    
    try:
        # Text report
        text_path = CONFIG.dir_reports / f"FORAI_{case_id}_{timestamp}.txt"
        write_text_report(text_path, header, qa)
        outputs["text"] = str(text_path)
        
        # JSON report for programmatic access
        json_path = CONFIG.dir_reports / f"FORAI_{case_id}_{timestamp}.json"
        write_json_report(json_path, header, qa)
        outputs["json"] = str(json_path)
        
        # CSV report for spreadsheet analysis
        csv_path = CONFIG.dir_reports / f"FORAI_{case_id}_{timestamp}.csv"
        write_csv_report(csv_path, header, qa)
        outputs["csv"] = str(csv_path)
        
        # PDF report if available
        if HAVE_PDF:
            try:
                pdf_path = CONFIG.dir_reports / f"FORAI_{case_id}_{timestamp}.pdf"
                write_pdf_report(pdf_path, header, qa)
                outputs["pdf"] = str(pdf_path)
            except Exception as e:
                LOGGER.warning(f"PDF generation failed: {e}")
        
        LOGGER.info(f"Generated {len(outputs)} report formats")
        
    except Exception as e:
        LOGGER.error(f"Report generation failed: {e}")
        raise
    
    return outputs

def write_text_report(path: Path, header: Dict[str, str], qa: List[Dict[str, Any]]):
    """Write enhanced text report"""
    
    with path.open('w', encoding='utf-8') as f:
        f.write("FORAI FORENSIC ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Header information
        f.write("CASE INFORMATION:\n")
        f.write("-" * 20 + "\n")
        for key, value in header.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Executive summary statistics
        total_evidence = sum(qa_item.get('evidence_count', 0) for qa_item in qa)
        avg_confidence = sum(qa_item.get('confidence', 0) for qa_item in qa) / len(qa) if qa else 0
        
        f.write("ANALYSIS SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Questions analyzed: {len(qa)}\n")
        f.write(f"Total evidence entries: {total_evidence}\n")
        f.write(f"Average confidence: {avg_confidence:.1%}\n")
        f.write("\n")
        
        # Question and answers
        f.write("FORENSIC QUESTIONS AND ANSWERS:\n")
        f.write("-" * 40 + "\n\n")
        
        for qa_item in qa:
            f.write(f"Q{qa_item['question_num']}: {qa_item['question']}\n")
            f.write(f"A{qa_item['question_num']}: {qa_item['answer']}\n")
            
            # Metadata
            confidence = qa_item.get('confidence', 0.0)
            evidence_count = qa_item.get('evidence_count', 0)
            f.write(f"[Confidence: {confidence:.1%} | Evidence: {evidence_count} entries]\n\n")
        
        # Footer
        f.write("-" * 50 + "\n")
        f.write(f"Report generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write("Generated by New_FORAI.py - Enhanced Forensic Analysis Tool\n")

def write_json_report(path: Path, header: Dict[str, str], qa: List[Dict[str, Any]]):
    """Write JSON report for programmatic access"""
    
    report_data = {
        "metadata": {
            "tool": "New_FORAI.py",
            "version": "2.0",
            "generated": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0"
        },
        "case_info": header,
        "analysis_summary": {
            "questions_count": len(qa),
            "total_evidence": sum(qa_item.get('evidence_count', 0) for qa_item in qa),
            "average_confidence": sum(qa_item.get('confidence', 0) for qa_item in qa) / len(qa) if qa else 0
        },
        "questions_and_answers": qa
    }
    
    with path.open('w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

def write_csv_report(path: Path, header: Dict[str, str], qa: List[Dict[str, Any]]):
    """Write CSV report for spreadsheet analysis"""
    
    if not HAVE_PANDAS:
        LOGGER.warning("Pandas not available, skipping CSV report")
        return
    
    # Prepare data for CSV
    csv_data = []
    for qa_item in qa:
        csv_data.append({
            "Question_Number": qa_item['question_num'],
            "Question": qa_item['question'],
            "Answer": qa_item['answer'],
            "Confidence": qa_item.get('confidence', 0.0),
            "Evidence_Count": qa_item.get('evidence_count', 0),
            "Sources": '; '.join(qa_item.get('sources', []))
        })
    
    df = pd.DataFrame(csv_data)
    df.to_csv(path, index=False, encoding='utf-8')

def write_pdf_report(path: Path, header: Dict[str, str], qa: List[Dict[str, Any]]):
    """Write PDF report"""
    
    if not HAVE_PDF:
        raise RuntimeError("FPDF not available for PDF generation")
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    
    # Title
    pdf.cell(0, 10, "FORAI FORENSIC ANALYSIS REPORT", 0, 1, "C")
    pdf.ln(10)
    
    # Case information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "CASE INFORMATION", 0, 1)
    pdf.set_font("Arial", "", 10)
    
    for key, value in header.items():
        pdf.cell(0, 8, f"{key}: {value}", 0, 1)
    
    pdf.ln(5)
    
    # Questions and answers
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "FORENSIC ANALYSIS", 0, 1)
    
    for qa_item in qa:
        pdf.set_font("Arial", "B", 10)
        question_text = f"Q{qa_item['question_num']}: {qa_item['question']}"
        pdf.multi_cell(0, 8, question_text)
        
        pdf.set_font("Arial", "", 10)
        answer_text = f"A{qa_item['question_num']}: {qa_item['answer']}"
        pdf.multi_cell(0, 8, answer_text)
        
        # Metadata
        confidence = qa_item.get('confidence', 0.0)
        evidence_count = qa_item.get('evidence_count', 0)
        pdf.set_font("Arial", "I", 8)
        pdf.cell(0, 6, f"Confidence: {confidence:.1%} | Evidence: {evidence_count} entries", 0, 1)
        pdf.ln(3)
    
    pdf.output(str(path))

def write_chain_of_custody(case_id: str) -> Path:
    """Enhanced chain of custody logging"""
    
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    coc_path = CONFIG.dir_reports / f"FORAI_CoC_{case_id}_{timestamp}.txt"
    
    with coc_path.open('w', encoding='utf-8') as f:
        f.write("FORAI CHAIN OF CUSTODY LOG\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Case ID: {case_id}\n")
        f.write(f"Analysis Tool: New_FORAI.py v2.0\n")
        f.write(f"Analysis Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
        f.write(f"Analyst: System\n\n")
        
        # System information
        f.write("ANALYSIS ENVIRONMENT:\n")
        f.write("-" * 25 + "\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"Platform: {os.name}\n")
        
        if HAVE_PSUTIL:
            memory_info = check_memory_usage()
            f.write(f"System Memory: {memory_info['total']:.1f} GB total, {memory_info['used']:.1f} GB used\n")
        
        f.write("\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-" * 15 + "\n")
        f.write(f"Base Directory: {CONFIG.base_dir}\n")
        f.write(f"Database Path: {CONFIG.db_path}\n")
        f.write(f"Max Workers: {CONFIG.max_workers}\n")
        f.write(f"Batch Size: {CONFIG.batch_size}\n")
        f.write("\n")
        
        # Dependencies
        f.write("DEPENDENCIES:\n")
        f.write("-" * 13 + "\n")
        f.write(f"Pandas: {'Available' if HAVE_PANDAS else 'Missing'}\n")
        f.write(f"TQDM: {'Available' if HAVE_TQDM else 'Missing'}\n")
        f.write(f"FPDF: {'Available' if HAVE_PDF else 'Missing'}\n")
        f.write(f"Llama-CPP: {'Available' if HAVE_LLAMA else 'Missing'}\n")
        f.write(f"PSUtil: {'Available' if HAVE_PSUTIL else 'Missing'}\n")
        
        if MISSING_DEPS:
            f.write(f"\nMissing Dependencies: {', '.join(MISSING_DEPS)}\n")
        
        f.write("\n")
        
        # File integrity
        f.write("FILE INTEGRITY:\n")
        f.write("-" * 16 + "\n")
        
        try:
            script_hash = sha256_file(Path(__file__))
            f.write(f"Script Hash (SHA256): {script_hash}\n")
        except:
            f.write("Script Hash: Unable to calculate\n")
        
        if CONFIG.db_path.exists():
            try:
                db_hash = sha256_file(CONFIG.db_path)
                f.write(f"Database Hash (SHA256): {db_hash}\n")
            except:
                f.write("Database Hash: Unable to calculate\n")
        
        f.write("\n")
        f.write("END OF CHAIN OF CUSTODY LOG\n")
    
    return coc_path

def write_case_archive() -> Path:
    """Enhanced case archiving with better compression"""
    
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    archive_path = CONFIG.dir_archives / f"FORAI_Archive_{timestamp}.zip"
    
    # Ensure archives directory exists
    CONFIG.dir_archives.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        
        # Add reports
        if CONFIG.dir_reports.exists():
            for report_file in CONFIG.dir_reports.glob("FORAI_*"):
                if report_file.is_file():
                    arcname = f"reports/{report_file.name}"
                    zipf.write(report_file, arcname)
        
        # Add database if it exists and is not too large
        if CONFIG.db_path.exists():
            db_size = CONFIG.db_path.stat().st_size
            if db_size < 100 * 1024 * 1024:  # 100MB limit
                zipf.write(CONFIG.db_path, f"database/{CONFIG.db_path.name}")
            else:
                LOGGER.warning(f"Database too large for archive: {db_size / (1024*1024):.1f} MB")
        
        # Add configuration and logs
        config_info = {
            "base_dir": str(CONFIG.base_dir),
            "max_workers": CONFIG.max_workers,
            "batch_size": CONFIG.batch_size,
            "llm_settings": {
                "context_size": CONFIG.llm_context_size,
                "max_tokens": CONFIG.llm_max_tokens,
                "temperature": CONFIG.llm_temperature
            },
            "dependencies": {
                "pandas": HAVE_PANDAS,
                "tqdm": HAVE_TQDM,
                "fpdf": HAVE_PDF,
                "llama_cpp": HAVE_LLAMA,
                "psutil": HAVE_PSUTIL
            },
            "missing_deps": MISSING_DEPS,
            "archive_created": datetime.now(timezone.utc).isoformat()
        }
        
        config_json = json.dumps(config_info, indent=2)
        zipf.writestr("config/analysis_config.json", config_json)
    
    LOGGER.info(f"Created case archive: {archive_path}")
    return archive_path

# =============================================================================
# MAIN FUNCTION - Enhanced workflow orchestration
# =============================================================================

def main():
    """
    Enhanced main orchestration function with better error handling,
    logging, and performance monitoring.
    """
    
    start_time = time.time()
    
    try:
        # Parse command line arguments with enhanced options
        parser = argparse.ArgumentParser(
            description="New_FORAI - Enhanced Forensic Analysis Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s --case-id CASE001 --mode ALL --use-llm
  %(prog)s --case-id CASE002 --mode BETWEEN --between 01012024-01312024
  %(prog)s --case-id CASE003 --mode DAYS_BEFORE --target 12312023 --days 7
            """
        )
        
        # Required arguments
        parser.add_argument('--case-id', required=True, 
                          help='Case identifier (required)')
        
        # Analysis scope
        parser.add_argument('--mode', 
                          choices=['ALL', 'BETWEEN', 'DAYS_BEFORE'], 
                          default='ALL',
                          help='Analysis time scope mode (default: ALL)')
        parser.add_argument('--between', 
                          help='Date range: MMDDYYYY-MMDDYYYY (for BETWEEN mode)')
        parser.add_argument('--target', 
                          help='Target date: MMDDYYYY (for DAYS_BEFORE mode)')
        parser.add_argument('--days', type=int, 
                          help='Number of days before target (for DAYS_BEFORE mode)')
        
        # Processing controls
        parser.add_argument('--no-ingest', action='store_true',
                          help='Skip CSV ingestion (reuse existing database)')
        parser.add_argument('--extracts-dir', 
                          default=str(CONFIG.dir_extracts),
                          help=f'Path to extracts directory (default: {CONFIG.dir_extracts})')
        
        # KAPE controls
        parser.add_argument('--target-drive', default='C:',
                          help='Target drive letter (default: C:)')
        parser.add_argument('--skip-collect', action='store_true',
                          help='Skip KAPE collection step')
        parser.add_argument('--skip-parse', action='store_true',
                          help='Skip KAPE parsing step')
        parser.add_argument('--skip-kape', action='store_true',
                          help='Skip all KAPE steps')
        
        # Enhanced LLM support
        parser.add_argument('--use-llm', action='store_true',
                          help='Generate LLM executive summary')
        parser.add_argument('--llm-model',
                          default=str(CONFIG.dir_llm / 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'),
                          help='Path to GGUF model file')
        parser.add_argument('--llm-max-tokens', type=int, 
                          default=CONFIG.llm_max_tokens,
                          help=f'Max tokens for LLM (default: {CONFIG.llm_max_tokens})')
        parser.add_argument('--ask', 
                          help='Ad-hoc natural language question')
        
        # Performance and logging
        parser.add_argument('--max-workers', type=int, 
                          default=CONFIG.max_workers,
                          help=f'Max parallel workers (default: {CONFIG.max_workers})')
        parser.add_argument('--batch-size', type=int, 
                          default=CONFIG.batch_size,
                          help=f'Database batch size (default: {CONFIG.batch_size})')
        parser.add_argument('--log-level', 
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          default='INFO',
                          help='Logging level (default: INFO)')
        parser.add_argument('--log-file',
                          help='Optional log file path')
        
        args = parser.parse_args()
        
        # Update configuration from arguments
        CONFIG.max_workers = args.max_workers
        CONFIG.batch_size = args.batch_size
        
        # Setup enhanced logging
        log_file = Path(args.log_file) if args.log_file else None
        global LOGGER
        LOGGER = setup_logging(args.log_level, log_file)
        
        LOGGER.info("=" * 60)
        LOGGER.info("New_FORAI - Enhanced Forensic Analysis Tool Starting")
        LOGGER.info("=" * 60)
        LOGGER.info(f"Case ID: {args.case_id}")
        LOGGER.info(f"Analysis Mode: {args.mode}")
        LOGGER.info(f"Configuration: {CONFIG.max_workers} workers, {CONFIG.batch_size} batch size")
        
        # Validate configuration
        config_issues = CONFIG.validate()
        if config_issues:
            LOGGER.warning("Configuration issues found:")
            for issue in config_issues:
                LOGGER.warning(f"  - {issue}")
        
        # Check dependencies
        if MISSING_DEPS:
            LOGGER.warning(f"Missing optional dependencies: {', '.join(MISSING_DEPS)}")
            LOGGER.info("Install missing dependencies for full functionality:")
            LOGGER.info("  pip install pandas tqdm fpdf llama-cpp-python psutil")
        
        # Initialize environment
        LOGGER.info("Initializing environment...")
        ensure_dirs()
        
        # Handle custom extracts directory
        if args.extracts_dir and Path(args.extracts_dir) != CONFIG.dir_extracts:
            CONFIG.base_dir = Path(args.extracts_dir).parent
            LOGGER.info(f"Using custom extracts directory: {args.extracts_dir}")
        
        # KAPE collection and parsing phase
        kape_success = True
        if not (args.skip_kape or (args.skip_collect and args.skip_parse)):
            try:
                LOGGER.info("Starting KAPE operations...")
                check_kape_prereqs()
                
                # Validate target drive
                tdrive = args.target_drive.rstrip('\\/')
                if len(tdrive) >= 2 and tdrive[1] == ':':
                    tsource = tdrive + '\\'
                else:
                    raise ValueError(f'Invalid target drive format: {args.target_drive}')
                
                is_live = tdrive.upper().startswith('C:')
                LOGGER.info(f"Target source: {tsource} (Live system: {is_live})")
                
                # Collection phase
                if not args.skip_collect:
                    LOGGER.info(f"KAPE collection: {tsource} -> {CONFIG.dir_artifacts}")
                    kape_collect(tsource)
                    LOGGER.info("KAPE collection completed")
                
                # Parsing phase
                if not args.skip_parse:
                    LOGGER.info(f"KAPE parsing: {CONFIG.dir_artifacts} -> {CONFIG.dir_extracts}")
                    kape_parse()
                    LOGGER.info("KAPE parsing completed")
                
                # Live system supplements
                if is_live and not args.skip_collect:
                    LOGGER.info("Running live system supplements...")
                    run_live_only_supplements()
                    LOGGER.info("Live system supplements completed")
                
            except Exception as e:
                LOGGER.error(f"KAPE operations failed: {e}")
                kape_success = False
                # Continue with analysis if extracts exist
                if not CONFIG.dir_extracts.exists() or not list(CONFIG.dir_extracts.glob("*.csv")):
                    LOGGER.error("No extracts available for analysis")
                    return 1
        else:
            LOGGER.info("KAPE operations skipped")
        
        # Copy setupapi logs
        try:
            LOGGER.info("Copying setupapi logs...")
            copy_setupapi_logs()
        except Exception as e:
            LOGGER.warning(f"setupapi copy failed: {e}")
        
        # Administrator privileges check
        if os.name == 'nt':
            try:
                import ctypes
                if not bool(ctypes.windll.shell32.IsUserAnAdmin()):
                    LOGGER.info("TIP: Run as Administrator for full access on live systems")
            except Exception:
                pass
        
        # Database initialization and ingestion phase
        LOGGER.info("Initializing database connection...")
        con = db_connect()
        
        try:
            # Initialize database schema
            initialize_database(con)
            
            # Ingest setupapi text logs
            try:
                LOGGER.info("Ingesting setupapi logs...")
                # This function needs to be implemented or imported
                # ingest_setupapi_text(con, args.case_id)
            except Exception as e:
                LOGGER.warning(f"setupapi ingestion failed: {e}")
            
            # CSV ingestion phase
            if not args.no_ingest:
                LOGGER.info(f"Starting parallel CSV ingestion from {CONFIG.dir_extracts}")
                memory_before = check_memory_usage()
                LOGGER.info(f"Memory before ingestion: {memory_before['percent']:.1%} used")
                
                total_rows = ingest_extracts_parallel(con, args.case_id, Path(args.extracts_dir))
                
                memory_after = check_memory_usage()
                LOGGER.info(f"Memory after ingestion: {memory_after['percent']:.1%} used")
                LOGGER.info(f"Ingestion completed: {total_rows:,} total rows")
            else:
                LOGGER.info("Skipping CSV ingestion (using existing database)")
            
            # Analysis scope and question answering phase
            LOGGER.info("Computing analysis scope...")
            start_epoch, end_epoch, range_text = compute_range(
                args.mode, args.between, args.target, args.days, con
            )
            
            LOGGER.info("Setting analysis scope...")
            set_analysis_scope(con, start_epoch, end_epoch, range_text)
            
            LOGGER.info("Extracting case header information...")
            header = get_header_values(con, range_text)
            
            LOGGER.info("Answering forensic questions...")
            qa_start_time = time.time()
            qa = answer_questions(con)
            qa_duration = time.time() - qa_start_time
            LOGGER.info(f"Question answering completed in {qa_duration:.1f}s")
            
            # Standard report generation
            LOGGER.info("Generating standard reports...")
            outputs = write_outputs(args.case_id, header, qa)
            
            # Chain of custody and archiving
            try:
                LOGGER.info("Creating chain of custody log...")
                coc_path = write_chain_of_custody(args.case_id)
                outputs["custody"] = str(coc_path)
            except Exception as e:
                LOGGER.error(f"Chain of custody creation failed: {e}")
            
            try:
                LOGGER.info("Creating case archive...")
                archive_path = write_case_archive()
                outputs["archive"] = str(archive_path)
            except Exception as e:
                LOGGER.error(f"Case archive creation failed: {e}")
            
            # Display standard outputs
            LOGGER.info("Standard outputs generated:")
            for output_type, path in outputs.items():
                if path:
                    LOGGER.info(f"  {output_type}: {path}")
            
            # Enhanced LLM executive summary
            if args.use_llm:
                if not HAVE_LLAMA:
                    LOGGER.error("LLM functionality requested but llama-cpp-python not available")
                elif not Path(args.llm_model).exists():
                    LOGGER.error(f"LLM model not found: {args.llm_model}")
                else:
                    try:
                        LOGGER.info("Generating enhanced LLM executive summary...")
                        llm_start_time = time.time()
                        
                        summary = llm_summarize_enhanced(
                            header, qa, args.llm_model, args.llm_max_tokens
                        )
                        
                        llm_duration = time.time() - llm_start_time
                        LOGGER.info(f"LLM summary generated in {llm_duration:.1f}s")
                        
                        # Save LLM summary
                        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                        llm_path = CONFIG.dir_reports / f"FORAI_{args.case_id}_{timestamp}_enhanced_llm.txt"
                        
                        with llm_path.open('w', encoding='utf-8') as f:
                            f.write("FORAI ENHANCED EXECUTIVE SUMMARY\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(summary)
                        
                        LOGGER.info(f"Enhanced LLM summary: {llm_path}")
                        
                    except Exception as e:
                        LOGGER.error(f"LLM summary generation failed: {e}")
            
            # Enhanced ad-hoc question
            if args.ask:
                if not HAVE_LLAMA:
                    LOGGER.error("LLM functionality requested but llama-cpp-python not available")
                elif not Path(args.llm_model).exists():
                    LOGGER.error(f"LLM model not found: {args.llm_model}")
                else:
                    try:
                        LOGGER.info(f"Processing ad-hoc question: {args.ask}")
                        ask_start_time = time.time()
                        
                        answer = llm_answer_custom_enhanced(
                            con, header, args.ask, args.llm_model, 
                            max_tokens=max(800, args.llm_max_tokens)
                        )
                        
                        ask_duration = time.time() - ask_start_time
                        LOGGER.info(f"Ad-hoc question answered in {ask_duration:.1f}s")
                        
                        # Save ad-hoc answer
                        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                        ask_path = CONFIG.dir_reports / f"FORAI_{args.case_id}_{timestamp}_enhanced_ask.txt"
                        
                        with ask_path.open('w', encoding='utf-8') as f:
                            f.write("FORAI ENHANCED AD-HOC ANALYSIS\n")
                            f.write("=" * 50 + "\n\n")
                            f.write(f"QUESTION: {args.ask}\n\n")
                            f.write("ANALYSIS:\n")
                            f.write(answer)
                        
                        LOGGER.info(f"Ad-hoc analysis: {ask_path}")
                        
                    except Exception as e:
                        LOGGER.error(f"Ad-hoc question processing failed: {e}")
        
        finally:
            # Cleanup database connection
            con.close()
            
            # Close connection pool
            with _connection_lock:
                for conn in _connection_pool.values():
                    try:
                        conn.close()
                    except:
                        pass
                _connection_pool.clear()
        
        # Final summary
        total_duration = time.time() - start_time
        LOGGER.info("=" * 60)
        LOGGER.info("ANALYSIS COMPLETE")
        LOGGER.info("=" * 60)
        LOGGER.info(f"Total execution time: {total_duration:.1f}s")
        LOGGER.info(f"KAPE operations: {'Success' if kape_success else 'Failed/Skipped'}")
        LOGGER.info(f"Questions answered: {len(qa) if 'qa' in locals() else 0}")
        LOGGER.info(f"Reports generated: {len(outputs) if 'outputs' in locals() else 0}")
        
        if 'memory_after' in locals():
            LOGGER.info(f"Final memory usage: {memory_after['percent']:.1%}")
        
        LOGGER.info("Enhanced accuracy prioritized over token economy.")
        
        return 0
        
    except KeyboardInterrupt:
        LOGGER.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        LOGGER.error(f"Fatal error during analysis: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())