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
# ENHANCED SEARCH SYSTEM FOR IMPROVED TINYLLAMA ACCURACY
# =============================================================================

class EnhancedForensicSearch:
    """Advanced FTS5 search system optimized for TinyLLama forensic analysis"""
    
    def __init__(self):
        self.artifact_weights = {
            'registry': 1.5,
            'filesystem': 1.3,
            'network': 1.4,
            'process': 1.2,
            'usb': 1.6,
            'browser': 1.1,
            'email': 1.4,
            'system': 1.2
        }
        
        self.forensic_expansions = {
            'exfiltration': ['copy', 'transfer', 'usb', 'upload', 'email', 'download'],
            'malware': ['virus', 'trojan', 'suspicious', 'executable', 'infection', 'payload'],
            'intrusion': ['login', 'access', 'authentication', 'breach', 'unauthorized'],
            'deletion': ['delete', 'remove', 'wipe', 'shred', 'format', 'destroy'],
            'modification': ['edit', 'change', 'alter', 'update', 'write', 'modify'],
            'communication': ['email', 'chat', 'message', 'call', 'contact', 'skype'],
            'storage': ['usb', 'drive', 'disk', 'volume', 'mount', 'device'],
            'network': ['internet', 'connection', 'traffic', 'packet', 'protocol'],
            'user': ['account', 'login', 'session', 'profile', 'authentication']
        }
    
    def enhanced_search_evidence(self, query: str, limit: int = 15) -> List[Dict]:
        """Multi-stage enhanced search with intelligent ranking"""
        
        # Stage 1: Query expansion with forensic keywords
        expanded_queries = self._expand_forensic_keywords(query)
        
        # Stage 2: Multi-query search with weighting
        all_results = []
        
        with get_database_connection() as conn:
            for expanded_query, weight in expanded_queries:
                results = self._weighted_fts_search(conn, expanded_query, weight, limit * 2)
                all_results.extend(results)
        
        # Stage 3: Remove duplicates and merge scores
        merged_results = self._merge_duplicate_results(all_results)
        
        # Stage 4: Temporal clustering
        clustered_results = self._cluster_by_time(merged_results)
        
        # Stage 5: Evidence correlation
        correlated_results = self._correlate_evidence(clustered_results)
        
        # Stage 6: Intelligent final ranking
        final_results = self._intelligent_ranking(correlated_results, query)
        
        return final_results[:limit]
    
    def _expand_forensic_keywords(self, query: str) -> List[Tuple[str, float]]:
        """Expand query with forensic-specific synonyms and related terms"""
        
        expanded_queries = [(query, 1.0)]  # Original query with highest weight
        query_lower = query.lower()
        
        # Add forensic domain expansions
        for key, expansions in self.forensic_expansions.items():
            if key in query_lower:
                for expansion in expansions[:3]:  # Limit to top 3 expansions
                    if expansion not in query_lower:
                        expanded_queries.append((f"({query}) OR {expansion}", 0.7))
        
        # Add common forensic patterns
        if any(term in query_lower for term in ['suspicious', 'anomaly', 'unusual']):
            expanded_queries.append((f"({query}) OR (anomalous OR irregular)", 0.6))
        
        return expanded_queries[:4]  # Limit total expansions
    
    def _weighted_fts_search(self, conn: sqlite3.Connection, query: str, weight: float, limit: int) -> List[Dict]:
        """FTS5 search with artifact type weighting and BM25 ranking"""
        
        try:
            results = conn.execute("""
                SELECT 
                    e.*,
                    bm25(evidence_fts, 1.0, 1.0, 1.0) as base_score,
                    ? as query_weight
                FROM evidence e
                JOIN evidence_fts ON evidence_fts.rowid = e.id
                WHERE evidence_fts MATCH ?
                ORDER BY bm25(evidence_fts) DESC
                LIMIT ?
            """, (weight, query, limit)).fetchall()
            
            # Convert to dictionaries and apply artifact weighting
            weighted_results = []
            for row in results:
                result_dict = dict(row)
                artifact_type = result_dict.get('artifact_type', '').lower()
                
                # Apply artifact-specific weighting
                artifact_weight = self.artifact_weights.get(artifact_type, 1.0)
                result_dict['weighted_score'] = result_dict['base_score'] * artifact_weight * weight
                
                weighted_results.append(result_dict)
            
            return weighted_results
            
        except sqlite3.OperationalError:
            # Fallback to basic search if FTS5 fails
            return self._basic_search_fallback(conn, query, limit)
    
    def _basic_search_fallback(self, conn: sqlite3.Connection, query: str, limit: int) -> List[Dict]:
        """Fallback search when FTS5 is not available"""
        
        results = conn.execute("""
            SELECT * FROM evidence 
            WHERE summary LIKE ? OR details LIKE ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (f'%{query}%', f'%{query}%', limit)).fetchall()
        
        return [dict(row) for row in results]
    
    def _merge_duplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Merge duplicate results and combine their scores"""
        
        merged = {}
        
        for result in results:
            result_id = result.get('id')
            if result_id:
                if result_id in merged:
                    # Combine scores for duplicate results
                    merged[result_id]['weighted_score'] += result.get('weighted_score', 0)
                    merged[result_id]['query_matches'] = merged[result_id].get('query_matches', 1) + 1
                else:
                    result['query_matches'] = 1
                    merged[result_id] = result
        
        return list(merged.values())
    
    def _cluster_by_time(self, results: List[Dict]) -> List[Dict]:
        """Group evidence by temporal proximity for better context"""
        
        time_clusters = defaultdict(list)
        
        for result in results:
            if result.get('timestamp'):
                # Group by hour for temporal correlation
                dt = datetime.fromtimestamp(result['timestamp'])
                hour_key = dt.replace(minute=0, second=0, microsecond=0)
                time_clusters[hour_key].append(result)
        
        # Boost scores for items in clusters with multiple evidence
        clustered_results = []
        for cluster_time, cluster_items in time_clusters.items():
            cluster_boost = min(len(cluster_items) * 0.1, 0.4)  # Max 40% boost
            
            for item in cluster_items:
                item['temporal_score'] = item.get('weighted_score', 0) + cluster_boost
                item['cluster_size'] = len(cluster_items)
                item['cluster_time'] = cluster_time
                clustered_results.append(item)
        
        return clustered_results
    
    def _correlate_evidence(self, results: List[Dict]) -> List[Dict]:
        """Find correlations between different evidence types"""
        
        # Group by user and host for correlation analysis
        user_host_groups = defaultdict(list)
        
        for result in results:
            key = (result.get('username', ''), result.get('hostname', ''))
            user_host_groups[key].append(result)
        
        # Boost scores for evidence from same user/host
        correlated_results = []
        for (user, host), group_items in user_host_groups.items():
            if len(group_items) > 1:  # Multiple evidence from same source
                correlation_boost = min(len(group_items) * 0.12, 0.5)
                
                for item in group_items:
                    item['correlation_score'] = item.get('temporal_score', 0) + correlation_boost
                    item['correlation_count'] = len(group_items)
                    correlated_results.append(item)
            else:
                # Single evidence, no correlation boost
                item = group_items[0]
                item['correlation_score'] = item.get('temporal_score', 0)
                item['correlation_count'] = 1
                correlated_results.append(item)
        
        return correlated_results
    
    def _intelligent_ranking(self, results: List[Dict], original_query: str) -> List[Dict]:
        """Final intelligent ranking considering multiple factors"""
        
        # Extract key terms from original query for relevance scoring
        query_terms = set(re.findall(r'\w+', original_query.lower()))
        
        for result in results:
            # Calculate term relevance score
            content = f"{result.get('summary', '')} {result.get('details', '')}".lower()
            content_terms = set(re.findall(r'\w+', content))
            
            term_overlap = len(query_terms.intersection(content_terms))
            relevance_score = term_overlap / max(len(query_terms), 1)
            
            # Final composite score
            result['final_ranking_score'] = (
                result.get('correlation_score', 0) * 0.4 +    # Correlation weight
                relevance_score * 0.25 +                      # Relevance weight
                (result.get('cluster_size', 1) / 10) * 0.15 + # Temporal clustering
                result.get('query_matches', 1) * 0.1 +        # Multi-query matches
                result.get('weighted_score', 0) * 0.1         # Original FTS5 score
            )
        
        # Sort by final ranking score
        return sorted(results, key=lambda x: x.get('final_ranking_score', 0), reverse=True)
    
    def build_optimized_context(self, results: List[Dict], max_tokens: int = 1800) -> str:
        """Build optimized context for TinyLLama within token limits"""
        
        context_parts = []
        current_tokens = 0
        seen_types = set()
        
        for result in results:
            artifact_type = result.get('artifact_type', '')
            
            # Prefer diverse evidence types for better context
            type_penalty = 0.1 if artifact_type in seen_types else 0
            adjusted_score = result.get('final_ranking_score', 0) - type_penalty
            
            if adjusted_score > 0.2:  # Quality threshold
                # Build concise evidence summary
                timestamp_str = ""
                if result.get('timestamp'):
                    dt = datetime.fromtimestamp(result['timestamp'])
                    timestamp_str = f"[{dt.strftime('%m/%d %H:%M')}] "
                
                # Create concise but informative summary
                summary = result.get('summary', '')[:90]
                evidence_text = f"{timestamp_str}{artifact_type.upper()}: {summary}"
                
                # Add correlation info if significant
                if result.get('correlation_count', 1) > 2:
                    evidence_text += f" (correlated with {result['correlation_count']} events)"
                
                # Estimate tokens (rough: 4 chars per token)
                estimated_tokens = len(evidence_text) // 4
                
                if current_tokens + estimated_tokens < max_tokens:
                    context_parts.append(evidence_text)
                    current_tokens += estimated_tokens
                    seen_types.add(artifact_type)
                else:
                    break
        
        return "\n".join(context_parts)

# Global instance for enhanced search
enhanced_search = EnhancedForensicSearch()

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

def calculate_file_hash(file_path: Path) -> str:
    """Calculate file hash using streaming to handle large files"""
    hash_obj = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

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
        file_hash = calculate_file_hash(file_path)
        
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
        """Advanced response validation - less restrictive for forensic analysis"""
        if not response or len(response) < 10:
            return False
        
        # Check for clear hallucination indicators (more permissive for forensic language)
        hallucination_patterns = [
            r"I believe", r"I think", r"in my opinion", r"I assume",
            r"I guess", r"I suppose", r"I imagine"
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
        """Enhanced forensic question answering with improved TinyLLama accuracy"""
        
        # Use enhanced search system for better evidence retrieval
        evidence_results = enhanced_search.enhanced_search_evidence(question, limit=20)
        
        if not evidence_results:
            return "Insufficient evidence in scope."
        
        # Build optimized context for TinyLLama
        optimized_context = enhanced_search.build_optimized_context(evidence_results, max_tokens=1800)
        
        # Enhanced prompt structure for better TinyLLama performance
        enhanced_prompt = f"""FORENSIC ANALYSIS TASK:
Question: {question}

EVIDENCE (chronological, correlated):
{optimized_context}

ANALYSIS INSTRUCTIONS:
- Focus on temporal patterns and correlations between evidence
- Identify key artifacts and their relationships
- Provide specific evidence-based conclusions
- Note any suspicious activity patterns or anomalies
- Consider user behavior and system interactions

FORENSIC RESPONSE:"""
        
        # Try LLM analysis with enhanced context
        if self.llm.llm:
            llm_response = self.llm.generate_response(question, enhanced_prompt)
            if llm_response not in ["LLM not available", "Error generating response", "Response failed validation checks"]:
                return llm_response
        
        # Enhanced fallback with structured analysis
        return self._generate_enhanced_structured_analysis(evidence_results, question)
    
    def _generate_enhanced_structured_analysis(self, evidence_results: List[Dict], question: str) -> str:
        """Generate enhanced structured analysis when LLM unavailable"""
        
        # Analyze evidence patterns for better insights
        artifact_counts = defaultdict(int)
        time_range = {"earliest": None, "latest": None}
        users = set()
        hosts = set()
        correlations = defaultdict(list)
        
        for result in evidence_results:
            # Count artifact types
            artifact_type = result.get('artifact_type', 'unknown')
            artifact_counts[artifact_type] += 1
            
            # Track time range
            if result.get('timestamp'):
                ts = result['timestamp']
                if not time_range["earliest"] or ts < time_range["earliest"]:
                    time_range["earliest"] = ts
                if not time_range["latest"] or ts > time_range["latest"]:
                    time_range["latest"] = ts
            
            # Track users and hosts
            if result.get('username'):
                users.add(result['username'])
            if result.get('hostname'):
                hosts.add(result['hostname'])
            
            # Track correlations
            if result.get('correlation_count', 0) > 1:
                key = f"{result.get('username', 'Unknown')}@{result.get('hostname', 'Unknown')}"
                correlations[key].append(result)
        
        # Generate enhanced structured response
        analysis = f"ENHANCED FORENSIC ANALYSIS - {question}\n"
        analysis += "=" * 60 + "\n\n"
        
        analysis += "EVIDENCE SUMMARY:\n"
        analysis += f"• Total artifacts analyzed: {len(evidence_results)}\n"
        analysis += f"• Artifact types: {', '.join(sorted(artifact_counts.keys()))}\n"
        analysis += f"• Users involved: {', '.join(sorted(users)) if users else 'Unknown'}\n"
        analysis += f"• Hosts involved: {', '.join(sorted(hosts)) if hosts else 'Unknown'}\n"
        
        if time_range["earliest"] and time_range["latest"]:
            start_time = datetime.fromtimestamp(time_range["earliest"])
            end_time = datetime.fromtimestamp(time_range["latest"])
            duration = end_time - start_time
            analysis += f"• Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}\n"
            analysis += f"• Duration: {duration}\n"
        
        analysis += "\nKEY FINDINGS (ranked by relevance):\n"
        for i, result in enumerate(evidence_results[:8], 1):
            score = result.get('final_ranking_score', 0)
            cluster_info = f" (clustered with {result.get('cluster_size', 1)} events)" if result.get('cluster_size', 1) > 1 else ""
            correlation_info = f" (correlated with {result.get('correlation_count', 1)} events)" if result.get('correlation_count', 1) > 1 else ""
            
            timestamp_str = ""
            if result.get('timestamp'):
                dt = datetime.fromtimestamp(result['timestamp'])
                timestamp_str = f"[{dt.strftime('%m/%d %H:%M')}] "
            
            analysis += f"{i}. {timestamp_str}{result.get('artifact_type', 'UNKNOWN').upper()}: "
            analysis += f"{result.get('summary', 'No summary available')[:100]}"
            analysis += f"{cluster_info}{correlation_info} (score: {score:.2f})\n"
        
        # Add correlation analysis if significant correlations found
        if correlations:
            analysis += "\nCORRELATION ANALYSIS:\n"
            for user_host, correlated_events in correlations.items():
                if len(correlated_events) > 2:
                    analysis += f"• {user_host}: {len(correlated_events)} correlated events detected\n"
        
        # Add pattern analysis
        analysis += "\nPATTERN ANALYSIS:\n"
        if 'usb' in artifact_counts and artifact_counts['usb'] > 1:
            analysis += "• Multiple USB device activities detected - potential data transfer\n"
        if 'network' in artifact_counts and 'filesystem' in artifact_counts:
            analysis += "• Network and filesystem activity correlation - potential exfiltration pattern\n"
        if len(users) > 1:
            analysis += f"• Multiple user accounts involved ({len(users)} users) - potential privilege escalation\n"
        
        return analysis

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
    
    def _sanitize_text_for_pdf(self, text: str) -> str:
        """Sanitize text for PDF generation to handle Unicode issues"""
        if not text:
            return ""
        
        # Convert to string and handle None values
        text = str(text)
        
        # Replace problematic Unicode characters with ASCII equivalents
        replacements = {
            '\u2013': '-',  # en dash
            '\u2014': '--', # em dash
            '\u2018': "'",  # left single quote
            '\u2019': "'",  # right single quote
            '\u201c': '"',  # left double quote
            '\u201d': '"',  # right double quote
            '\u2026': '...' # ellipsis
        }
        
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        
        # Remove or replace other non-ASCII characters
        try:
            text.encode('latin1')
            return text
        except UnicodeEncodeError:
            # Replace non-encodable characters with '?'
            return text.encode('latin1', errors='replace').decode('latin1')

    def _generate_pdf_report(self, report: Dict[str, Any], output_path: Path):
        """Generate PDF report using modern FPDF with Unicode handling"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        
        title = self._sanitize_text_for_pdf(f'Forensic Analysis Report - Case {self.case_id}')
        pdf.cell(0, 10, title, 0, 1, 'C')
        
        pdf.set_font('Arial', '', 12)
        pdf.ln(10)
        
        # Computer Identity
        if report['computer_identity']:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'Computer Identity', 0, 1)
            pdf.set_font('Arial', '', 12)
            
            for key, value in report['computer_identity'].items():
                if value:
                    clean_key = self._sanitize_text_for_pdf(key.replace("_", " ").title())
                    clean_value = self._sanitize_text_for_pdf(str(value))
                    pdf.cell(0, 8, f'{clean_key}: {clean_value}', 0, 1)
            pdf.ln(5)
        
        # User Accounts
        if report['user_accounts']:
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, 'User Accounts', 0, 1)
            pdf.set_font('Arial', '', 12)
            
            for account in report['user_accounts'][:5]:  # Top 5 accounts
                username = self._sanitize_text_for_pdf(account['username'])
                activity = self._sanitize_text_for_pdf(str(account['activity_count']))
                pdf.cell(0, 8, f"User: {username} (Activity: {activity})", 0, 1)
            pdf.ln(5)
        
        # Forensic Answers
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Forensic Analysis', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        for question, answer in report['forensic_answers'].items():
            pdf.set_font('Arial', 'B', 11)
            clean_question = self._sanitize_text_for_pdf(question)
            pdf.multi_cell(0, 6, f'Q: {clean_question}', 0, 1)
            
            pdf.set_font('Arial', '', 10)
            clean_answer = self._sanitize_text_for_pdf(answer)
            pdf.multi_cell(0, 5, f'A: {clean_answer}', 0, 1)
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