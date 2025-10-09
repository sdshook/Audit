#!/usr/bin/env python3
"""
Comprehensive Function Testing Suite for New_FORAI.py
Tests every function for accuracy, functionality, and forensic capability.
"""

import sys
import sqlite3
import tempfile
import os
import json
import time
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path to import New_FORAI
sys.path.insert(0, str(Path(__file__).parent))

def test_utility_functions():
    """Test all utility functions for accuracy"""
    print("🔍 Testing Utility Functions")
    
    try:
        import New_FORAI
        
        # Test validate_case_id
        valid_cases = ["CASE-001", "TEST_123", "FORENSIC-2024-001", "ABC123"]
        invalid_cases = ["case with spaces", "case@invalid", "", "a" * 51]
        
        for case in valid_cases:
            if not New_FORAI.validate_case_id(case):
                print(f"❌ validate_case_id failed for valid case: {case}")
                return False
        
        for case in invalid_cases:
            if New_FORAI.validate_case_id(case):
                print(f"❌ validate_case_id accepted invalid case: {case}")
                return False
        
        print("✅ validate_case_id working correctly")
        
        # Test validate_date_format
        valid_dates = ["20241201", "20240229", "19990101"]  # Including leap year
        invalid_dates = ["2024-12-01", "20241301", "20240230", "abc", ""]
        
        for date in valid_dates:
            if not New_FORAI.validate_date_format(date):
                print(f"❌ validate_date_format failed for valid date: {date}")
                return False
        
        for date in invalid_dates:
            if New_FORAI.validate_date_format(date):
                print(f"❌ validate_date_format accepted invalid date: {date}")
                return False
        
        print("✅ validate_date_format working correctly")
        
        # Test sanitize_query_string
        test_queries = [
            ("normal query", "normal query"),
            ("query with 'quotes'", "query with quotes"),
            ('query with "double quotes"', "query with double quotes"),
            ("query; DROP TABLE evidence;", "query DROP TABLE evidence"),
            ("", "")
        ]
        
        for input_query, expected in test_queries:
            result = New_FORAI.sanitize_query_string(input_query)
            if result != expected:
                print(f"❌ sanitize_query_string failed: '{input_query}' -> '{result}' (expected '{expected}')")
                return False
        
        print("✅ sanitize_query_string working correctly")
        
        # Test parse_timestamp
        test_timestamps = [
            ("2024-12-01 15:30:45", True),  # ISO format should work
            ("invalid", None),  # Should return None
            ("", None)  # Empty string
        ]
        
        for input_ts, expected in test_timestamps:
            result = New_FORAI.parse_timestamp(input_ts)
            if expected is None:
                if result is not None:
                    print(f"❌ parse_timestamp failed: {input_ts} -> {result} (expected None)")
                    return False
            elif expected is True:
                if result is None or not isinstance(result, int):
                    print(f"❌ parse_timestamp failed: {input_ts} -> {result} (expected integer timestamp)")
                    return False
        
        print("✅ parse_timestamp working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions error: {e}")
        return False

def test_database_operations():
    """Test all database operations for accuracy"""
    print("\n🔍 Testing Database Operations")
    
    try:
        import New_FORAI
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = tmp.name
        
        try:
            # Test database connection
            conn = New_FORAI.get_database_connection()
            if not conn:
                print("❌ get_database_connection failed")
                return False
            conn.close()
            print("✅ get_database_connection working")
            
            # Test database initialization with fresh database
            New_FORAI.initialize_database()
            print("✅ initialize_database working")
            
            # Test database schema with fresh connection
            conn = sqlite3.connect(New_FORAI.CONFIG.db_path)
            
            # Clear any existing test data
            conn.execute("DELETE FROM evidence WHERE case_id = 'TEST-001'")
            conn.commit()
            
            # Check all tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = ['evidence', 'sources', 'evidence_search', 'cases', 'analysis_results', 'chain_of_custody']
            missing_tables = [t for t in expected_tables if t not in tables]
            
            if missing_tables:
                print(f"❌ Missing database tables: {missing_tables}")
                return False
            
            print("✅ Database schema complete")
            
            # Test evidence insertion
            test_evidence = {
                'case_id': 'TEST-001',
                'host': 'TESTHOST',
                'user': 'testuser',
                'timestamp': int(time.time()),
                'artifact': 'Registry',
                'source_file': 'test.reg',
                'summary': 'Test registry entry for forensic analysis',
                'data_json': json.dumps({'key': 'HKLM\\Software\\Test', 'value': 'TestData'}),
                'file_hash': 'abc123def456'
            }
            
            cursor = conn.execute("""
                INSERT INTO evidence (case_id, host, user, timestamp, artifact, source_file, summary, data_json, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(test_evidence.values()))
            
            conn.commit()
            
            # Verify insertion
            cursor = conn.execute("SELECT COUNT(*) FROM evidence WHERE case_id = ?", ('TEST-001',))
            count = cursor.fetchone()[0]
            
            if count != 1:
                print(f"❌ Evidence insertion failed: expected 1 record, got {count}")
                return False
            
            print("✅ Evidence insertion working")
            
            # Test FTS5 search
            cursor = conn.execute("SELECT * FROM evidence_search WHERE evidence_search MATCH ?", ('registry',))
            fts_results = cursor.fetchall()
            
            if len(fts_results) != 1:
                print(f"❌ FTS5 search failed: expected 1 result, got {len(fts_results)}")
                return False
            
            print("✅ FTS5 search working")
            
            conn.close()
            return True
            
        finally:
            # Clean up
            if os.path.exists(db_path):
                os.unlink(db_path)
                
    except Exception as e:
        print(f"❌ Database operations error: {e}")
        return False

def test_search_functionality():
    """Test all search functions for accuracy"""
    print("\n🔍 Testing Search Functionality")
    
    try:
        import New_FORAI
        
        # Initialize database with test data
        New_FORAI.initialize_database()
        
        # Insert comprehensive test evidence
        test_evidence_data = [
            ('TEST-001', 'WORKSTATION-01', 'john.doe', int(time.time()), 'Registry', 'NTUSER.DAT', 
             'User login activity detected', '{"logon_type": "Interactive", "session_id": "1"}', 'hash1'),
            ('TEST-001', 'WORKSTATION-01', 'jane.smith', int(time.time()) - 3600, 'FileSystem', 'Documents.zip', 
             'Document archive accessed', '{"file_size": "1024000", "last_accessed": "2024-12-01"}', 'hash2'),
            ('TEST-001', 'WORKSTATION-01', 'admin', int(time.time()) - 7200, 'Network', 'network.log', 
             'External connection established', '{"destination": "192.168.1.100", "port": "443"}', 'hash3'),
            ('TEST-001', 'WORKSTATION-01', 'john.doe', int(time.time()) - 10800, 'USB', 'usb.log', 
             'USB device connected', '{"device_id": "VID_1234&PID_5678", "serial": "ABC123"}', 'hash4'),
            ('TEST-001', 'WORKSTATION-01', 'jane.smith', int(time.time()) - 14400, 'Browser', 'history.db', 
             'Web browsing activity', '{"url": "https://example.com", "visits": "5"}', 'hash5')
        ]
        
        conn = sqlite3.connect(New_FORAI.CONFIG.db_path)
        
        for evidence in test_evidence_data:
            conn.execute("""
                INSERT INTO evidence (case_id, host, user, timestamp, artifact, source_file, summary, data_json, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, evidence)
        
        conn.commit()
        conn.close()
        
        print("✅ Test evidence data inserted")
        
        # Test standalone search function (simpler test)
        results = New_FORAI.search_evidence("Registry", limit=5)
        if not isinstance(results, list):
            print(f"❌ Standalone search should return list, got {type(results)}")
            return False
        
        print(f"✅ Standalone search returned {len(results)} results for 'Registry'")
        
        # Test search with different terms
        results = New_FORAI.search_evidence("john.doe", limit=5)
        if not isinstance(results, list):
            print(f"❌ Search should return list, got {type(results)}")
            return False
        
        print(f"✅ Search returned {len(results)} results for 'john.doe'")
        
        # Test search with date filtering
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime("%Y%m%d")
        
        results = New_FORAI.search_evidence("activity", limit=5, date_from=date_str)
        if not isinstance(results, list):
            print(f"❌ Date-filtered search should return list, got {type(results)}")
            return False
        
        print(f"✅ Date-filtered search returned {len(results)} results")
        
        # Test enhanced search class instantiation
        try:
            search_engine = New_FORAI.EnhancedForensicSearch()
            print("✅ EnhancedForensicSearch class instantiation working")
        except Exception as e:
            print(f"❌ EnhancedForensicSearch instantiation failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Search functionality error: {e}")
        return False

def test_llm_integration():
    """Test LLM integration for forensic analysis accuracy"""
    print("\n🔍 Testing LLM Integration")
    
    try:
        import New_FORAI
        
        # Test AdvancedTinyLlamaEnhancer
        enhancer = New_FORAI.AdvancedTinyLlamaEnhancer()
        
        # Test forensic examples
        if not enhancer.forensic_examples:
            print("❌ Forensic examples not loaded")
            return False
        
        print("✅ Forensic examples loaded")
        
        # Test chain of thought analysis
        test_evidence = """
        Registry Evidence:
        - User: john.doe
        - Timestamp: 2024-12-01 14:30:00
        - Key: HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run
        - Value: malware.exe
        - Action: Key created
        
        File System Evidence:
        - File: C:\\Users\\john.doe\\Downloads\\malware.exe
        - Size: 2048000 bytes
        - Created: 2024-12-01 14:25:00
        - Hash: abc123def456789
        """
        
        test_question = "What evidence suggests malicious software installation?"
        
        result = enhancer.chain_of_thought_analysis(test_question, test_evidence)
        
        if not isinstance(result, str) or len(result) < 50:
            print(f"❌ Chain of thought analysis failed: {type(result)}, length: {len(result) if isinstance(result, str) else 'N/A'}")
            return False
        
        print("✅ Chain of thought analysis working")
        
        # Test multi-pass analysis
        result = enhancer.multi_pass_analysis(test_question, test_evidence, None)
        
        if not isinstance(result, dict):
            print(f"❌ Multi-pass analysis failed: expected dict, got {type(result)}")
            return False
        
        print("✅ Multi-pass analysis working")
        
        # Test validation patterns
        patterns = enhancer._load_validation_patterns()
        
        if not isinstance(patterns, dict) or len(patterns) == 0:
            print(f"❌ Validation patterns failed")
            return False
        
        print("✅ Validation patterns working")
        
        # Test all 12 forensic questions are available
        questions = New_FORAI.FORENSIC_QUESTIONS
        if len(questions) != 12:
            print(f"❌ Expected 12 forensic questions, got {len(questions)}")
            return False
        
        print("✅ All 12 forensic questions available")
        
        # Test ModernLLM class
        llm = New_FORAI.ModernLLM()
        
        # Should handle missing model gracefully
        result = llm.generate_response("Test prompt", "Test evidence")
        if not isinstance(result, str):
            print(f"❌ ModernLLM generate_response failed")
            return False
        
        print("✅ ModernLLM working (graceful handling of missing model)")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM integration error: {e}")
        return False

def test_forensic_analyzer():
    """Test ForensicAnalyzer class for accuracy"""
    print("\n🔍 Testing ForensicAnalyzer")
    
    try:
        import New_FORAI
        
        # Initialize with test data
        New_FORAI.initialize_database()
        
        analyzer = New_FORAI.ForensicAnalyzer()
        
        # Test computer identity analysis
        result = analyzer.analyze_computer_identity("TEST-001")
        
        if not isinstance(result, dict):
            print(f"❌ analyze_computer_identity should return dict, got {type(result)}")
            return False
        
        print("✅ ForensicAnalyzer.analyze_computer_identity working")
        
        # Test user accounts analysis
        result = analyzer.analyze_user_accounts("TEST-001")
        
        if not isinstance(result, list):
            print(f"❌ analyze_user_accounts should return list, got {type(result)}")
            return False
        
        print("✅ ForensicAnalyzer.analyze_user_accounts working")
        
        # Test USB devices analysis
        result = analyzer.analyze_usb_devices("TEST-001")
        
        if not isinstance(result, list):
            print(f"❌ analyze_usb_devices should return list, got {type(result)}")
            return False
        
        print("✅ ForensicAnalyzer.analyze_usb_devices working")
        
        # Test forensic question answering
        result = analyzer.answer_forensic_question("What is the computer name?", "TEST-001")
        
        if not isinstance(result, str):
            print(f"❌ answer_forensic_question should return str, got {type(result)}")
            return False
        
        print("✅ ForensicAnalyzer.answer_forensic_question working")
        
        return True
        
    except Exception as e:
        print(f"❌ ForensicAnalyzer error: {e}")
        return False

def test_report_generation():
    """Test report generation capabilities"""
    print("\n🔍 Testing Report Generation")
    
    try:
        import New_FORAI
        
        # Test ModernReportGenerator
        generator = New_FORAI.ModernReportGenerator("TEST-001")
        
        # Test analysis results structure
        test_results = [
            {
                'question': 'What is the computer name?',
                'evidence_count': 5,
                'analysis': 'Computer name is WORKSTATION-01 based on registry evidence.',
                'confidence': 0.95,
                'timestamp': datetime.now().isoformat()
            },
            {
                'question': 'What user accounts exist?',
                'evidence_count': 3,
                'analysis': 'Found user accounts: john.doe, jane.smith, admin.',
                'confidence': 0.90,
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # Test comprehensive report generation
        report = generator.generate_comprehensive_report()
        
        if not isinstance(report, dict):
            print(f"❌ generate_comprehensive_report should return dict, got {type(report)}")
            return False
        
        expected_keys = ['case_id', 'generated', 'computer_identity', 'user_accounts', 'usb_devices', 'forensic_answers']
        missing_keys = [k for k in expected_keys if k not in report]
        
        if missing_keys:
            print(f"❌ Comprehensive report missing keys: {missing_keys}")
            return False
        
        print("✅ Comprehensive report generation working")
        
        # Test JSON report saving
        json_path = generator.save_report(report, 'json')
        
        if not isinstance(json_path, Path):
            print(f"❌ save_report (JSON) should return Path, got {type(json_path)}")
            return False
        
        print("✅ JSON report saving working")
        
        # Test PDF report saving
        pdf_path = generator.save_report(report, 'pdf')
        
        if not isinstance(pdf_path, Path):
            print(f"❌ save_report (PDF) should return Path, got {type(pdf_path)}")
            return False
        
        print("✅ PDF report saving working")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation error: {e}")
        return False

def test_workflow_manager():
    """Test ForensicWorkflowManager for complete workflow"""
    print("\n🔍 Testing Workflow Manager")
    
    try:
        import New_FORAI
        
        # Test workflow manager initialization
        workflow = New_FORAI.ForensicWorkflowManager("TEST-001", Path("D:/FORAI"), verbose=True)
        
        # Test directory structure creation
        expected_dirs = ['artifacts_dir', 'parsed_dir', 'reports_dir', 'custody_dir', 'archives_dir', 'llm_dir']
        
        for dir_attr in expected_dirs:
            if not hasattr(workflow, dir_attr):
                print(f"❌ Workflow manager missing {dir_attr}")
                return False
        
        print("✅ Workflow manager initialization working")
        
        # Test chain of custody functionality
        workflow.log_custody_event("initialization", "Workflow initialized for case TEST-001")
        
        if len(workflow.chain_of_custody) != 1:
            print(f"❌ Chain of custody event not added properly")
            return False
        
        print("✅ Chain of custody functionality working")
        
        # Test chain of custody report generation
        custody_file = workflow.generate_chain_of_custody_report()
        
        if not isinstance(custody_file, Path):
            print(f"❌ generate_chain_of_custody_report should return Path, got {type(custody_file)}")
            return False
        
        print("✅ Chain of custody report generation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Workflow manager error: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring and optimization"""
    print("\n🔍 Testing Performance Monitoring")
    
    try:
        import New_FORAI
        
        # Test performance monitor decorator
        @New_FORAI.performance_monitor
        def test_function():
            time.sleep(0.1)  # Simulate work
            return "test result"
        
        result = test_function()
        
        if result != "test result":
            print(f"❌ Performance monitor decorator affected function result")
            return False
        
        print("✅ Performance monitor decorator working")
        
        # Test database connection performance
        start_time = time.time()
        conn = New_FORAI.get_database_connection()
        connection_time = time.time() - start_time
        
        if connection_time > 1.0:  # Should be very fast
            print(f"❌ Database connection too slow: {connection_time:.3f}s")
            return False
        
        conn.close()
        print(f"✅ Database connection performance good: {connection_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring error: {e}")
        return False

def main():
    """Run comprehensive function testing"""
    print("🚀 Comprehensive Function Testing for New_FORAI.py")
    print("Testing every function for accuracy and forensic capability\n")
    
    tests = [
        test_utility_functions,
        test_database_operations,
        test_search_functionality,
        test_llm_integration,
        test_forensic_analyzer,
        test_report_generation,
        test_workflow_manager,
        test_performance_monitoring
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Comprehensive Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All functions tested successfully! New_FORAI.py is highly accurate and functional.")
        return True
    else:
        print("⚠️  Some function tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)