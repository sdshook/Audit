#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New_FORAI.py (c) 2025 All Rights Reserved Shane D. Shook
Modern forensic analysis tool - Maximum efficiency and accuracy
Zero backward compatibility - Modern Python only

DESIGN PRINCIPLES:
- Maximum accuracy through advanced algorithms
- Peak efficiency via modern Python patterns  
- Zero backward compatibility - modern only
- Required dependencies for full functionality
- Streamlined codebase with no legacy support

=============================================================================
"""

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
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Set, Union, Any, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from functools import lru_cache, wraps

# Required imports - fail fast if not available
import pandas as pd
from tqdm import tqdm
from fpdf import FPDF
from llama_cpp import Llama
import psutil

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ForaiConfig:
    """Modern configuration for maximum performance"""
    
    base_dir: Path = Path("./FORAI")
    max_workers: int = min(8, (os.cpu_count() or 4))
    batch_size: int = 10000
    chunk_size: int = 50000
    memory_threshold: float = 0.85
    
    # LLM settings - optimized for accuracy
    llm_context_size: int = 16384
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.01
    llm_top_p: float = 0.9
    llm_threads: int = min(8, (os.cpu_count() or 4))
    
    # Database settings
    db_wal_mode: bool = True
    db_cache_size: int = 50000
    db_mmap_size: int = 1073741824  # 1GB
    
    def __post_init__(self):
        """Initialize directories"""
        for subdir in ["archives", "artifacts", "extracts", "llm", "reports", "tools"]:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    @property
    def db_path(self) -> Path:
        return self.base_dir / "extracts" / "forai.db"

CONFIG = ForaiConfig()

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    """Setup structured logging"""
    logger = logging.getLogger("FORAI")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger

LOGGER = setup_logging()

# =============================================================================
# CONSTANTS
# =============================================================================

SYSTEM_PROMPT = """You are an expert digital forensics analyst. Follow these rules for maximum accuracy:

1. EVIDENCE-ONLY: Base ALL statements strictly on provided evidence. Never invent or assume facts.
2. INSUFFICIENT DATA: If evidence is insufficient, state exactly: "Insufficient evidence in scope."
3. NEUTRAL LANGUAGE: Use factual, non-accusatory language. Avoid speculation about intent.
4. STRUCTURED OUTPUT: Use clear bullets with timestamps, filenames, and user accounts when available.
5. LIMITATIONS: Always note evidence limitations and distinguish correlation from causation.
"""

ARTIFACT_PATTERNS = {
    re.compile(r"systeminfo|system_info", re.I): "SystemInfo",
    re.compile(r"setupapi|setup_api", re.I): "SetupAPI", 
    re.compile(r"storage|disk|physicaldrive", re.I): "Storage",
    re.compile(r"sam_|sam\.", re.I): "SAM",
    re.compile(r"ntuser|nt_user", re.I): "NTUSER",
    re.compile(r"event.*logon|logon.*event|4624|4647|4634", re.I): "EventLogon",
    re.compile(r"usbstor|usb_stor", re.I): "USBStorage",
    re.compile(r"mountpoints|mount_points", re.I): "MountPoints",
    re.compile(r"mft|master_file_table|\$mft", re.I): "MFT",
    re.compile(r"usnjrnl|usn_journal|\$j", re.I): "USNJournal",
    re.compile(r"jumplist|jump_list|lecmd", re.I): "JumpList",
    re.compile(r"browser|history|chrome|firefox|edge", re.I): "BrowserHistory",
    re.compile(r"dns|domain_name", re.I): "DNS",
    re.compile(r"process|task|proc", re.I): "Process",
    re.compile(r"filesystem|file_system|shellbag", re.I): "FileSystem",
    re.compile(r"print|spool", re.I): "Print",
    re.compile(r"amcache|am_cache", re.I): "AmCache",
    re.compile(r"service|svc", re.I): "Services",
    re.compile(r"prefetch|pf", re.I): "Prefetch",
    re.compile(r"registry|reg", re.I): "Registry",
    re.compile(r"network|net|tcp|udp", re.I): "Network",
}

TIME_COLUMNS = {
    "TimeCreated", "EventCreatedTime", "Timestamp", "TimeStamp", "Created", 
    "CreationTime", "LastAccess", "LastWrite", "LastWriteTime", "FirstRun", 
    "LastRun", "RecordCreateTime", "FileCreated", "FileModified", "Modified", 
    "WriteTime", "Accessed", "AccessedTime", "ModifiedTime", "ExecutionTime", 
    "InstallTime", "UninstallTime", "StartTime", "EndTime", "LogonTime", "LogoffTime"
}

FORENSIC_QUESTIONS = [
    "What is the computer name, make, model, and serial number?",
    "What are the internal storage devices (make, model, serial numbers)?", 
    "What user accounts exist with their SIDs and activity timeframes?",
    "Who is the primary user based on activity volume and recency?",
    "Is there evidence of anti-forensic activities (log clearing, file deletion, timestamp modification)?",
    "What removable storage devices were connected (make, model, serial, timeframes)?",
    "What files were transferred to/from removable storage devices?",
    "What cloud storage services were accessed and what files were transferred?",
    "Were screenshots or screen recordings created?",
    "What documents were printed and when?",
    "What software was installed, uninstalled, or modified?",
    "What network connections and communications occurred?"
]

# =============================================================================
# DATABASE SCHEMA
# =============================================================================

DATABASE_SCHEMA = """
-- Core evidence table - optimized structure
CREATE TABLE IF NOT EXISTS evidence (
    id          INTEGER PRIMARY KEY,
    case_id     TEXT NOT NULL,
    host        TEXT,
    user        TEXT,
    timestamp   INTEGER,
    artifact    TEXT NOT NULL,
    source_file TEXT NOT NULL,
    summary     TEXT,
    data_json   TEXT,
    file_hash   TEXT,
    created     INTEGER DEFAULT (unixepoch())
) STRICT;

-- Source files tracking
CREATE TABLE IF NOT EXISTS sources (
    file_path   TEXT PRIMARY KEY,
    file_hash   TEXT,
    file_size   INTEGER,
    processed   INTEGER DEFAULT (unixepoch()),
    status      TEXT DEFAULT 'complete'
) STRICT;

-- Analysis scope for current session
CREATE TABLE IF NOT EXISTS scope (
    start_time  INTEGER,
    end_time    INTEGER,
    description TEXT
) STRICT;

-- High-performance indexes
CREATE INDEX IF NOT EXISTS idx_evidence_time ON evidence(timestamp) WHERE timestamp IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_evidence_artifact ON evidence(artifact);
CREATE INDEX IF NOT EXISTS idx_evidence_user ON evidence(user) WHERE user IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_evidence_host ON evidence(host) WHERE host IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_evidence_case ON evidence(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_composite ON evidence(case_id, artifact, timestamp);

-- Advanced full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_search USING fts5(
    summary, data_json,
    content='evidence',
    content_rowid='id',
    tokenize='trigram'
);

-- Auto-sync FTS triggers
CREATE TRIGGER IF NOT EXISTS sync_fts_insert AFTER INSERT ON evidence BEGIN
    INSERT INTO evidence_search(rowid, summary, data_json) 
    VALUES (new.id, COALESCE(new.summary, ''), COALESCE(new.data_json, '{}'));
END;

CREATE TRIGGER IF NOT EXISTS sync_fts_delete AFTER DELETE ON evidence BEGIN
    INSERT INTO evidence_search(evidence_search, rowid, summary, data_json) 
    VALUES('delete', old.id, old.summary, old.data_json);
END;

CREATE TRIGGER IF NOT EXISTS sync_fts_update AFTER UPDATE ON evidence BEGIN
    INSERT INTO evidence_search(evidence_search, rowid, summary, data_json) 
    VALUES('delete', old.id, old.summary, old.data_json);
    INSERT INTO evidence_search(rowid, summary, data_json) 
    VALUES (new.id, new.summary, new.data_json);
END;
"""

# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def performance_monitor(func):
    """Performance monitoring decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            LOGGER.debug(f"{func.__name__}: {end_time - start_time:.2f}s, "
                        f"Memory: {end_memory - start_memory:+.1f}MB")
    
    return wrapper

@performance_monitor
def get_database_connection() -> sqlite3.Connection:
    """Get optimized database connection"""
    conn = sqlite3.connect(
        CONFIG.db_path,
        timeout=30.0,
        check_same_thread=False
    )
    
    # Modern SQLite optimizations
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA cache_size={CONFIG.db_cache_size}")
    conn.execute(f"PRAGMA mmap_size={CONFIG.db_mmap_size}")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA optimize")
    
    return conn

@lru_cache(maxsize=1000)
def parse_timestamp(timestamp_str: str) -> Optional[int]:
    """High-performance timestamp parsing with caching"""
    if not timestamp_str or timestamp_str.lower() in ('null', 'none', ''):
        return None
    
    # Modern timestamp formats (most common first)
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(timestamp_str.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue
    
    return None

@lru_cache(maxsize=500)
def detect_artifact_type(filename: str) -> str:
    """Fast artifact type detection"""
    filename_lower = filename.lower()
    
    for pattern, artifact_type in ARTIFACT_PATTERNS.items():
        if pattern.search(filename_lower):
            return artifact_type
    
    return "Unknown"

def extract_json_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize important forensic fields"""
    normalized = {}
    
    # User identification
    for key in ['User', 'UserName', 'Username', 'AccountName', 'SubjectUserName']:
        if key in data and data[key]:
            normalized['user'] = str(data[key])
            break
    
    # System identification  
    for key in ['ComputerName', 'Computer_Name', 'Hostname', 'Host']:
        if key in data and data[key]:
            normalized['host'] = str(data[key])
            break
    
    # File paths
    for key in ['FilePath', 'File_Path', 'Path', 'FullPath']:
        if key in data and data[key]:
            normalized['file_path'] = str(data[key])
            break
    
    return normalized

@performance_monitor
def initialize_database() -> None:
    """Initialize database with optimized schema"""
    with get_database_connection() as conn:
        conn.executescript(DATABASE_SCHEMA)
        conn.commit()
    
    LOGGER.info("Database initialized with optimized schema")

@performance_monitor
def process_csv_file(file_path: Path, case_id: str) -> int:
    """Process CSV file with modern pandas optimizations"""
    LOGGER.info(f"Processing CSV file: {file_path}")
    
    try:
        # Read CSV with optimizations
        df = pd.read_csv(
            file_path,
            dtype=str,
            na_filter=False,
            engine='c',
            low_memory=False,
            chunksize=CONFIG.chunk_size
        )
        
        total_rows = 0
        artifact_type = detect_artifact_type(file_path.name)
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        
        with get_database_connection() as conn:
            # Record source file
            conn.execute("""
                INSERT OR REPLACE INTO sources (file_path, file_hash, file_size)
                VALUES (?, ?, ?)
            """, (str(file_path), file_hash, file_path.stat().st_size))
            
            # Process in chunks
            for chunk in df:
                rows_to_insert = []
                
                for _, row in chunk.iterrows():
                    # Extract timestamp
                    timestamp = None
                    for col in TIME_COLUMNS:
                        if col in row and row[col]:
                            timestamp = parse_timestamp(row[col])
                            if timestamp:
                                break
                    
                    # Extract normalized fields
                    data_dict = row.to_dict()
                    normalized = extract_json_fields(data_dict)
                    
                    # Create summary
                    summary_parts = []
                    for key, value in data_dict.items():
                        if value and len(str(value)) < 100:
                            summary_parts.append(f"{key}: {value}")
                    summary = " | ".join(summary_parts[:5])  # Limit summary length
                    
                    rows_to_insert.append((
                        case_id,
                        normalized.get('host'),
                        normalized.get('user'),
                        timestamp,
                        artifact_type,
                        str(file_path),
                        summary,
                        json.dumps(data_dict, ensure_ascii=False),
                        file_hash
                    ))
                
                # Batch insert
                conn.executemany("""
                    INSERT INTO evidence 
                    (case_id, host, user, timestamp, artifact, source_file, summary, data_json, file_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, rows_to_insert)
                
                total_rows += len(rows_to_insert)
            
            conn.commit()
        
        LOGGER.info(f"Processed {total_rows} rows from {file_path}")
        return total_rows
        
    except Exception as e:
        LOGGER.error(f"Error processing {file_path}: {e}")
        return 0

@performance_monitor
def search_evidence(query: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Advanced full-text search with modern FTS5"""
    with get_database_connection() as conn:
        cursor = conn.execute("""
            SELECT e.id, e.case_id, e.host, e.user, e.timestamp, e.artifact,
                   e.source_file, e.summary, e.data_json,
                   rank
            FROM evidence_search 
            JOIN evidence e ON evidence_search.rowid = e.id
            WHERE evidence_search MATCH ?
            ORDER BY rank
            LIMIT ?
        """, (query, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'case_id': row[1],
                'host': row[2],
                'user': row[3],
                'timestamp': row[4],
                'artifact': row[5],
                'source_file': row[6],
                'summary': row[7],
                'data_json': json.loads(row[8]) if row[8] else {},
                'rank': row[9]
            })
        
        return results

# =============================================================================
# LLM INTEGRATION
# =============================================================================

class ModernLLM:
    """Modern LLM integration with advanced guardrails"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or CONFIG.base_dir / "llm" / "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize LLM with optimized settings"""
        if not self.model_path.exists():
            LOGGER.warning(f"LLM model not found at {self.model_path}")
            return
        
        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=CONFIG.llm_context_size,
                n_threads=CONFIG.llm_threads,
                verbose=False
            )
            LOGGER.info("LLM model initialized successfully")
        except Exception as e:
            LOGGER.error(f"Failed to initialize LLM: {e}")
    
    def generate_response(self, prompt: str, evidence: str) -> str:
        """Generate response with advanced guardrails"""
        if not self.llm:
            return "LLM not available"
        
        full_prompt = f"{SYSTEM_PROMPT}\n\nEvidence:\n{evidence}\n\nQuestion: {prompt}\n\nAnswer:"
        
        try:
            response = self.llm(
                full_prompt,
                max_tokens=CONFIG.llm_max_tokens,
                temperature=CONFIG.llm_temperature,
                top_p=CONFIG.llm_top_p,
                stop=["Question:", "Evidence:", "\n\n"],
                echo=False
            )
            
            answer = response['choices'][0]['text'].strip()
            
            # Advanced validation
            if self._validate_response(answer):
                return answer
            else:
                return "Response failed validation checks"
                
        except Exception as e:
            LOGGER.error(f"LLM generation error: {e}")
            return "Error generating response"
    
    def _validate_response(self, response: str) -> bool:
        """Advanced response validation"""
        if not response or len(response) < 10:
            return False
        
        # Check for hallucination indicators
        hallucination_patterns = [
            r"I believe", r"I think", r"probably", r"likely", r"seems like",
            r"appears to", r"suggests that", r"indicates that"
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, response, re.I):
                LOGGER.warning(f"Potential hallucination detected: {pattern}")
                return False
        
        return True

# =============================================================================
# FORENSIC ANALYSIS
# =============================================================================

class ForensicAnalyzer:
    """Modern forensic analysis engine"""
    
    def __init__(self):
        self.llm = ModernLLM()
    
    @performance_monitor
    def analyze_computer_identity(self, case_id: str) -> Dict[str, Any]:
        """Analyze computer identity with modern SQL"""
        with get_database_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT
                    json_extract(data_json, '$.ComputerName') as computer_name,
                    json_extract(data_json, '$.SystemManufacturer') as make,
                    json_extract(data_json, '$.SystemProductName') as model,
                    json_extract(data_json, '$.SystemSerialNumber') as serial,
                    COUNT(*) as evidence_count
                FROM evidence 
                WHERE case_id = ? 
                  AND (json_extract(data_json, '$.ComputerName') IS NOT NULL
                    OR json_extract(data_json, '$.SystemManufacturer') IS NOT NULL)
                GROUP BY computer_name, make, model, serial
                ORDER BY evidence_count DESC
                LIMIT 1
            """, (case_id,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'computer_name': result[0],
                    'make': result[1],
                    'model': result[2],
                    'serial': result[3],
                    'evidence_count': result[4]
                }
            
            return {}
    
    @performance_monitor
    def analyze_user_accounts(self, case_id: str) -> List[Dict[str, Any]]:
        """Analyze user accounts with modern aggregation"""
        with get_database_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    user,
                    json_extract(data_json, '$.SID') as sid,
                    MIN(timestamp) as first_activity,
                    MAX(timestamp) as last_activity,
                    COUNT(*) as activity_count,
                    COUNT(DISTINCT artifact) as artifact_types
                FROM evidence 
                WHERE case_id = ? 
                  AND user IS NOT NULL 
                  AND user != ''
                  AND user NOT LIKE '%$'
                GROUP BY user, sid
                HAVING activity_count > 5
                ORDER BY activity_count DESC
            """, (case_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'username': row[0],
                    'sid': row[1],
                    'first_activity': row[2],
                    'last_activity': row[3],
                    'activity_count': row[4],
                    'artifact_types': row[5]
                })
            
            return results
    
    @performance_monitor
    def analyze_usb_devices(self, case_id: str) -> List[Dict[str, Any]]:
        """Analyze USB devices with modern pattern matching"""
        with get_database_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT
                    json_extract(data_json, '$.DeviceManufacturer') as make,
                    json_extract(data_json, '$.DeviceModel') as model,
                    json_extract(data_json, '$.SerialNumber') as serial,
                    MIN(timestamp) as first_connected,
                    MAX(timestamp) as last_connected,
                    COUNT(*) as connection_count
                FROM evidence 
                WHERE case_id = ? 
                  AND artifact IN ('USBStorage', 'SetupAPI', 'MountPoints', 'Registry')
                  AND (summary LIKE '%usb%' OR summary LIKE '%removable%')
                GROUP BY make, model, serial
                ORDER BY first_connected DESC
            """, (case_id,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'make': row[0],
                    'model': row[1],
                    'serial': row[2],
                    'first_connected': row[3],
                    'last_connected': row[4],
                    'connection_count': row[5]
                })
            
            return results
    
    def answer_forensic_question(self, question: str, case_id: str) -> str:
        """Answer forensic question using modern analysis"""
        # Get relevant evidence
        evidence_results = search_evidence(question, limit=50)
        
        if not evidence_results:
            return "Insufficient evidence in scope."
        
        # Format evidence for LLM
        evidence_text = []
        for result in evidence_results[:10]:  # Limit to top 10 results
            timestamp_str = ""
            if result['timestamp']:
                dt = datetime.fromtimestamp(result['timestamp'], tz=timezone.utc)
                timestamp_str = f" [{dt.strftime('%Y-%m-%d %H:%M:%S UTC')}]"
            
            evidence_text.append(
                f"Artifact: {result['artifact']}{timestamp_str}\n"
                f"User: {result['user'] or 'Unknown'}\n"
                f"Host: {result['host'] or 'Unknown'}\n"
                f"Summary: {result['summary']}\n"
            )
        
        evidence_str = "\n---\n".join(evidence_text)
        
        # Generate response using LLM
        return self.llm.generate_response(question, evidence_str)

# =============================================================================
# REPORT GENERATION
# =============================================================================

class ModernReportGenerator:
    """Modern report generation with multiple formats"""
    
    def __init__(self, case_id: str):
        self.case_id = case_id
        self.analyzer = ForensicAnalyzer()
    
    @performance_monitor
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive forensic report"""
        report = {
            'case_id': self.case_id,
            'generated': datetime.now(timezone.utc).isoformat(),
            'computer_identity': self.analyzer.analyze_computer_identity(self.case_id),
            'user_accounts': self.analyzer.analyze_user_accounts(self.case_id),
            'usb_devices': self.analyzer.analyze_usb_devices(self.case_id),
            'forensic_answers': {}
        }
        
        # Answer all forensic questions
        for question in FORENSIC_QUESTIONS:
            answer = self.analyzer.answer_forensic_question(question, self.case_id)
            report['forensic_answers'][question] = answer
        
        return report
    
    def save_report(self, report: Dict[str, Any], format: str = 'json') -> Path:
        """Save report in specified format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format.lower() == 'json':
            report_path = CONFIG.base_dir / "reports" / f"forensic_report_{self.case_id}_{timestamp}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        elif format.lower() == 'pdf':
            report_path = CONFIG.base_dir / "reports" / f"forensic_report_{self.case_id}_{timestamp}.pdf"
            self._generate_pdf_report(report, report_path)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        LOGGER.info(f"Report saved: {report_path}")
        return report_path
    
    def _generate_pdf_report(self, report: Dict[str, Any], output_path: Path):
        """Generate PDF report using modern FPDF"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, f'Forensic Analysis Report - Case {self.case_id}', 0, 1, 'C')
        
        pdf.set_font('Arial', '', 12)
        pdf.ln(10)
        
        # Computer Identity
        if report['computer_identity']:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Computer Identity', 0, 1)
            pdf.set_font('Arial', '', 12)
            
            for key, value in report['computer_identity'].items():
                if value:
                    pdf.cell(0, 8, f'{key.replace("_", " ").title()}: {value}', 0, 1)
            pdf.ln(5)
        
        # User Accounts
        if report['user_accounts']:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'User Accounts', 0, 1)
            pdf.set_font('Arial', '', 12)
            
            for account in report['user_accounts'][:5]:  # Top 5 accounts
                pdf.cell(0, 8, f"User: {account['username']} (Activity: {account['activity_count']})", 0, 1)
            pdf.ln(5)
        
        # Forensic Answers
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Forensic Analysis', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        for question, answer in report['forensic_answers'].items():
            pdf.set_font('Arial', 'B', 11)
            pdf.multi_cell(0, 6, f'Q: {question}', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 5, f'A: {answer}', 0, 1)
            pdf.ln(3)
        
        pdf.output(str(output_path))

# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """Modern main workflow"""
    parser = argparse.ArgumentParser(
        description="Modern Forensic Analysis Tool - Maximum Efficiency & Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--case-id', required=True, help='Case identifier')
    parser.add_argument('--csv-dir', type=Path, help='Directory containing CSV files to process')
    parser.add_argument('--csv-file', type=Path, help='Single CSV file to process')
    parser.add_argument('--search', help='Search query for evidence')
    parser.add_argument('--question', help='Forensic question to answer')
    parser.add_argument('--report', choices=['json', 'pdf'], help='Generate comprehensive report')
    parser.add_argument('--init-db', action='store_true', help='Initialize database')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    
    try:
        # Initialize database if requested
        if args.init_db:
            initialize_database()
            return
        
        # Process CSV files
        if args.csv_dir:
            csv_files = list(args.csv_dir.glob('*.csv'))
            LOGGER.info(f"Found {len(csv_files)} CSV files to process")
            
            with ThreadPoolExecutor(max_workers=CONFIG.max_workers) as executor:
                futures = [
                    executor.submit(process_csv_file, csv_file, args.case_id)
                    for csv_file in csv_files
                ]
                
                total_rows = 0
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CSV files"):
                    total_rows += future.result()
            
            LOGGER.info(f"Processed {total_rows} total rows")
        
        elif args.csv_file:
            rows_processed = process_csv_file(args.csv_file, args.case_id)
            LOGGER.info(f"Processed {rows_processed} rows from {args.csv_file}")
        
        # Search evidence
        if args.search:
            results = search_evidence(args.search)
            print(f"\nFound {len(results)} results for: {args.search}")
            for result in results[:10]:
                print(f"- {result['artifact']}: {result['summary'][:100]}...")
        
        # Answer forensic question
        if args.question:
            analyzer = ForensicAnalyzer()
            answer = analyzer.answer_forensic_question(args.question, args.case_id)
            print(f"\nQuestion: {args.question}")
            print(f"Answer: {answer}")
        
        # Generate report
        if args.report:
            generator = ModernReportGenerator(args.case_id)
            report = generator.generate_comprehensive_report()
            report_path = generator.save_report(report, args.report)
            print(f"\nReport generated: {report_path}")
    
    except Exception as e:
        LOGGER.error(f"Error in main workflow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()