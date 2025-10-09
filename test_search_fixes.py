#!/usr/bin/env python3
"""
Test script to identify and fix search functionality issues in New_FORAI.py
"""

import tempfile
import os
import sqlite3
from pathlib import Path
import sys

# Add current directory to path to import New_FORAI
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import New_FORAI

def test_search_fixes():
    """Test and demonstrate fixes for search functionality"""
    
    print("Testing search functionality fixes...")
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override the CONFIG to use our temp directory
        New_FORAI.CONFIG.base_dir = Path(temp_dir)
        
        # Create the extracts subdirectory
        extracts_dir = Path(temp_dir) / 'extracts'
        extracts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        New_FORAI.initialize_database()
        
        # Check what FTS table was actually created
        with New_FORAI.get_database_connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%search%'")
            fts_tables = [row[0] for row in cursor.fetchall()]
            print(f"FTS tables found: {fts_tables}")
            
            # Add test data
            test_data = [
                ('CASE001', 'HOST1', 'user1', 1640995200, 'registry', 'test.reg', 'USB device connected', '{"device": "USB"}', 'hash1'),
                ('CASE001', 'HOST1', 'user1', 1640995300, 'filesystem', 'test.log', 'Malware file detected', '{"file": "malware.exe"}', 'hash2'),
                ('CASE001', 'HOST2', 'user2', 1640995400, 'network', 'network.pcap', 'Suspicious network activity', '{"ip": "192.168.1.100"}', 'hash3')
            ]
            
            conn.executemany('''
                INSERT INTO evidence (case_id, host, user, timestamp, artifact, source_file, summary, data_json, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', test_data)
            conn.commit()
            
            # Test the fallback search directly
            print("\nTesting fallback search...")
            try:
                # Set row_factory to get dict-like results
                conn.row_factory = sqlite3.Row
                results = conn.execute("""
                    SELECT * FROM evidence 
                    WHERE summary LIKE ? OR data_json LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, ('%USB%', '%USB%', 5)).fetchall()
                
                dict_results = [dict(row) for row in results]
                print(f"✓ Fallback search works: found {len(dict_results)} results")
                for result in dict_results:
                    print(f"  - {result['summary']}")
                    
            except Exception as e:
                print(f"✗ Fallback search failed: {e}")
                
            # Test FTS5 search if available
            if 'evidence_search' in fts_tables:
                print("\nTesting FTS5 search...")
                try:
                    results = conn.execute("""
                        SELECT e.*, bm25(evidence_search) as score
                        FROM evidence e
                        JOIN evidence_search ON evidence_search.rowid = e.id
                        WHERE evidence_search MATCH ?
                        ORDER BY bm25(evidence_search)
                        LIMIT ?
                    """, ('USB', 5)).fetchall()
                    
                    dict_results = [dict(row) for row in results]
                    print(f"✓ FTS5 search works: found {len(dict_results)} results")
                    for result in dict_results:
                        print(f"  - {result['summary']} (score: {result.get('score', 'N/A')})")
                        
                except Exception as e:
                    print(f"✗ FTS5 search failed: {e}")
            
        print("\nSearch functionality analysis completed!")

if __name__ == "__main__":
    test_search_fixes()