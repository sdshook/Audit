#!/usr/bin/env python3
"""
Forensic Accuracy Test for New_FORAI.py
Tests the tool's ability to answer all 12 forensic questions with high accuracy
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone
import sqlite3

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

import New_FORAI

def create_comprehensive_test_data():
    """Create comprehensive forensic test data covering all 12 question areas"""
    
    # Initialize database
    New_FORAI.initialize_database()
    
    # Clear existing test data
    with New_FORAI.get_database_connection() as conn:
        conn.execute("DELETE FROM evidence WHERE case_id = 'FORENSIC-TEST'")
        conn.commit()
    
    # Comprehensive test evidence covering all forensic areas
    test_evidence = [
        # Computer Identity Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'system',
            'timestamp': int(time.time()) - 86400,
            'artifact': 'Registry',
            'source_file': 'SYSTEM',
            'summary': 'Computer name and system information',
            'data_json': json.dumps({
                'ComputerName': 'WORKSTATION-ALPHA',
                'SystemManufacturer': 'Dell Inc.',
                'SystemProductName': 'OptiPlex 7090',
                'SystemSerialNumber': 'ABC123XYZ',
                'RegisteredOwner': 'Corporate IT',
                'RegisteredOrganization': 'ACME Corporation'
            }),
            'file_hash': 'hash_system_001'
        },
        
        # User Account Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'john.smith',
            'timestamp': int(time.time()) - 3600,
            'artifact': 'Registry',
            'source_file': 'SAM',
            'summary': 'User account information',
            'data_json': json.dumps({
                'Username': 'john.smith',
                'FullName': 'John Smith',
                'LastLogin': '2024-10-09 08:30:15',
                'AccountType': 'Administrator',
                'PasswordLastSet': '2024-09-15 14:22:33',
                'LoginCount': 247
            }),
            'file_hash': 'hash_sam_001'
        },
        
        # USB Device Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'john.smith',
            'timestamp': int(time.time()) - 7200,
            'artifact': 'Registry',
            'source_file': 'SYSTEM',
            'summary': 'USB device connection',
            'data_json': json.dumps({
                'DeviceType': 'USB Storage',
                'VendorID': '0951',
                'ProductID': '1666',
                'SerialNumber': '1A2B3C4D5E6F',
                'FriendlyName': 'Kingston DataTraveler 3.0',
                'FirstInstallDate': '2024-10-09 10:15:30',
                'LastConnected': '2024-10-09 14:22:15',
                'DriveLetter': 'E:'
            }),
            'file_hash': 'hash_usb_001'
        },
        
        # Network Activity Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'john.smith',
            'timestamp': int(time.time()) - 1800,
            'artifact': 'Network',
            'source_file': 'netstat.log',
            'summary': 'Network connection to external server',
            'data_json': json.dumps({
                'Protocol': 'TCP',
                'LocalAddress': '192.168.1.100',
                'LocalPort': '49152',
                'RemoteAddress': '203.0.113.45',
                'RemotePort': '443',
                'State': 'ESTABLISHED',
                'ProcessName': 'chrome.exe',
                'ProcessID': '2048'
            }),
            'file_hash': 'hash_network_001'
        },
        
        # File Activity Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'john.smith',
            'timestamp': int(time.time()) - 900,
            'artifact': 'FileSystem',
            'source_file': 'MFT',
            'summary': 'Suspicious file creation',
            'data_json': json.dumps({
                'FileName': 'confidential_data.zip',
                'FilePath': 'C:\\Users\\john.smith\\Desktop\\confidential_data.zip',
                'FileSize': 15728640,
                'CreatedTime': '2024-10-09 15:45:22',
                'ModifiedTime': '2024-10-09 15:47:18',
                'AccessedTime': '2024-10-09 15:47:18',
                'MD5Hash': 'a1b2c3d4e5f6789012345678901234ab',
                'SHA256Hash': 'abcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab'
            }),
            'file_hash': 'hash_file_001'
        },
        
        # Browser History Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'john.smith',
            'timestamp': int(time.time()) - 2700,
            'artifact': 'Browser',
            'source_file': 'History',
            'summary': 'Web browsing activity',
            'data_json': json.dumps({
                'URL': 'https://www.dropbox.com/upload',
                'Title': 'Upload Files - Dropbox',
                'VisitCount': 3,
                'LastVisitTime': '2024-10-09 15:15:33',
                'Browser': 'Chrome',
                'UserProfile': 'john.smith'
            }),
            'file_hash': 'hash_browser_001'
        },
        
        # Email Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'john.smith',
            'timestamp': int(time.time()) - 5400,
            'artifact': 'Email',
            'source_file': 'Outlook.pst',
            'summary': 'Email with attachment',
            'data_json': json.dumps({
                'From': 'john.smith@company.com',
                'To': 'external.contact@competitor.com',
                'Subject': 'Project Files',
                'SentTime': '2024-10-09 14:30:45',
                'HasAttachment': True,
                'AttachmentName': 'project_specs.docx',
                'AttachmentSize': 2048576,
                'MessageID': '<msg123@company.com>'
            }),
            'file_hash': 'hash_email_001'
        },
        
        # Application Usage Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'john.smith',
            'timestamp': int(time.time()) - 10800,
            'artifact': 'Application',
            'source_file': 'Prefetch',
            'summary': 'Application execution',
            'data_json': json.dumps({
                'ApplicationName': 'WINRAR.EXE',
                'ExecutionPath': 'C:\\Program Files\\WinRAR\\WinRAR.exe',
                'LastExecuted': '2024-10-09 12:45:30',
                'ExecutionCount': 15,
                'FilesAccessed': [
                    'C:\\Users\\john.smith\\Desktop\\confidential_data.zip',
                    'C:\\Users\\john.smith\\Documents\\archive.rar'
                ]
            }),
            'file_hash': 'hash_app_001'
        },
        
        # System Event Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'system',
            'timestamp': int(time.time()) - 14400,
            'artifact': 'EventLog',
            'source_file': 'Security.evtx',
            'summary': 'Failed login attempt',
            'data_json': json.dumps({
                'EventID': 4625,
                'LogonType': 3,
                'FailureReason': 'Unknown user name or bad password',
                'SourceIP': '192.168.1.200',
                'TargetUser': 'administrator',
                'WorkstationName': 'ATTACKER-PC',
                'LogonTime': '2024-10-09 11:30:15'
            }),
            'file_hash': 'hash_event_001'
        },
        
        # Deleted File Evidence
        {
            'case_id': 'FORENSIC-TEST',
            'host': 'WORKSTATION-ALPHA',
            'user': 'john.smith',
            'timestamp': int(time.time()) - 18000,
            'artifact': 'RecycleBin',
            'source_file': '$Recycle.Bin',
            'summary': 'Deleted sensitive file',
            'data_json': json.dumps({
                'OriginalFileName': 'employee_salaries.xlsx',
                'OriginalPath': 'C:\\Users\\john.smith\\Documents\\HR\\employee_salaries.xlsx',
                'DeletedTime': '2024-10-09 10:45:22',
                'FileSize': 524288,
                'RecycleBinPath': '$RECYCLE.BIN\\S-1-5-21-123456789-1234567890-123456789-1001\\$R1A2B3C.xlsx'
            }),
            'file_hash': 'hash_deleted_001'
        }
    ]
    
    # Insert test evidence
    with New_FORAI.get_database_connection() as conn:
        for evidence in test_evidence:
            conn.execute("""
                INSERT INTO evidence (case_id, host, user, timestamp, artifact, source_file, summary, data_json, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                evidence['case_id'], evidence['host'], evidence['user'], evidence['timestamp'],
                evidence['artifact'], evidence['source_file'], evidence['summary'],
                evidence['data_json'], evidence['file_hash']
            ))
        conn.commit()
    
    print(f"✅ Created {len(test_evidence)} comprehensive test evidence records")
    return len(test_evidence)

def test_forensic_question_accuracy():
    """Test the accuracy of answering all 12 forensic questions"""
    
    print("\n🔍 Testing Forensic Question Accuracy")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = New_FORAI.ForensicAnalyzer()
    
    # Test each forensic question
    results = {}
    total_score = 0
    max_score = len(New_FORAI.FORENSIC_QUESTIONS)
    
    for i, question in enumerate(New_FORAI.FORENSIC_QUESTIONS, 1):
        print(f"\n📋 Question {i}: {question}")
        
        try:
            # Get answer from analyzer
            start_time = time.time()
            answer = analyzer.answer_forensic_question(question, "FORENSIC-TEST")
            response_time = time.time() - start_time
            
            # Evaluate answer quality
            score = evaluate_answer_quality(question, answer)
            total_score += score
            
            results[question] = {
                'answer': answer,
                'score': score,
                'response_time': response_time,
                'length': len(answer) if answer else 0
            }
            
            print(f"   ⏱️  Response Time: {response_time:.2f}s")
            print(f"   📏 Answer Length: {len(answer) if answer else 0} characters")
            print(f"   ⭐ Quality Score: {score}/10")
            print(f"   💬 Answer Preview: {answer[:100] if answer else 'No answer'}...")
            
        except Exception as e:
            print(f"   ❌ Error answering question: {e}")
            results[question] = {
                'answer': None,
                'score': 0,
                'response_time': 0,
                'length': 0,
                'error': str(e)
            }
    
    # Calculate overall accuracy
    accuracy_percentage = (total_score / (max_score * 10)) * 100
    
    print(f"\n📊 Forensic Question Accuracy Results")
    print("=" * 60)
    print(f"Total Score: {total_score}/{max_score * 10}")
    print(f"Accuracy: {accuracy_percentage:.1f}%")
    
    if accuracy_percentage >= 80:
        print("🎉 EXCELLENT: High forensic accuracy achieved!")
    elif accuracy_percentage >= 60:
        print("✅ GOOD: Acceptable forensic accuracy")
    elif accuracy_percentage >= 40:
        print("⚠️  FAIR: Forensic accuracy needs improvement")
    else:
        print("❌ POOR: Forensic accuracy is insufficient")
    
    return results, accuracy_percentage

def evaluate_answer_quality(question: str, answer: str) -> int:
    """Evaluate the quality of a forensic answer (0-10 scale)"""
    
    if not answer or len(answer.strip()) < 10:
        return 0
    
    score = 0
    answer_lower = answer.lower()
    
    # Base score for having a substantial answer
    if len(answer) >= 50:
        score += 2
    
    # Check for specific forensic indicators based on question type
    if "computer" in question.lower() and "name" in question.lower():
        if "workstation-alpha" in answer_lower or "computer" in answer_lower:
            score += 3
        if "dell" in answer_lower or "optiplex" in answer_lower:
            score += 2
    
    elif "user" in question.lower() and "account" in question.lower():
        if "john.smith" in answer_lower or "administrator" in answer_lower:
            score += 3
        if "login" in answer_lower or "account" in answer_lower:
            score += 2
    
    elif "usb" in question.lower() or "device" in question.lower():
        if "kingston" in answer_lower or "datatraveler" in answer_lower:
            score += 3
        if "usb" in answer_lower or "storage" in answer_lower:
            score += 2
    
    elif "network" in question.lower() or "internet" in question.lower():
        if "203.0.113.45" in answer_lower or "443" in answer_lower:
            score += 3
        if "network" in answer_lower or "connection" in answer_lower:
            score += 2
    
    elif "file" in question.lower():
        if "confidential_data.zip" in answer_lower or "employee_salaries" in answer_lower:
            score += 3
        if "file" in answer_lower or "document" in answer_lower:
            score += 2
    
    elif "email" in question.lower():
        if "competitor.com" in answer_lower or "attachment" in answer_lower:
            score += 3
        if "email" in answer_lower or "message" in answer_lower:
            score += 2
    
    elif "application" in question.lower() or "program" in question.lower():
        if "winrar" in answer_lower or "chrome" in answer_lower:
            score += 3
        if "application" in answer_lower or "program" in answer_lower:
            score += 2
    
    elif "security" in question.lower() or "event" in question.lower():
        if "failed login" in answer_lower or "4625" in answer_lower:
            score += 3
        if "security" in answer_lower or "event" in answer_lower:
            score += 2
    
    # Additional quality indicators
    if "timestamp" in answer_lower or "time" in answer_lower:
        score += 1
    
    if "evidence" in answer_lower or "forensic" in answer_lower:
        score += 1
    
    # Cap at 10
    return min(score, 10)

def test_report_generation_quality():
    """Test the quality of forensic report generation"""
    
    print("\n📄 Testing Report Generation Quality")
    print("=" * 60)
    
    try:
        # Generate comprehensive report
        generator = New_FORAI.ModernReportGenerator("FORENSIC-TEST")
        
        start_time = time.time()
        report = generator.generate_comprehensive_report()
        generation_time = time.time() - start_time
        
        print(f"⏱️  Report Generation Time: {generation_time:.2f}s")
        
        # Evaluate report quality
        quality_score = 0
        max_quality = 50
        
        # Check report structure
        required_sections = ['case_id', 'generated', 'computer_identity', 'user_accounts', 'usb_devices', 'forensic_answers']
        for section in required_sections:
            if section in report:
                quality_score += 5
                print(f"✅ {section.replace('_', ' ').title()} section present")
            else:
                print(f"❌ {section.replace('_', ' ').title()} section missing")
        
        # Check forensic answers completeness
        if 'forensic_answers' in report:
            answered_questions = len(report['forensic_answers'])
            expected_questions = len(New_FORAI.FORENSIC_QUESTIONS)
            
            if answered_questions == expected_questions:
                quality_score += 10
                print(f"✅ All {expected_questions} forensic questions answered")
            else:
                partial_score = (answered_questions / expected_questions) * 10
                quality_score += partial_score
                print(f"⚠️  {answered_questions}/{expected_questions} forensic questions answered")
        
        # Check data richness
        if report.get('computer_identity'):
            quality_score += 5
            print("✅ Computer identity data present")
        
        if report.get('user_accounts'):
            quality_score += 5
            print("✅ User account data present")
        
        # Test report saving
        json_path = generator.save_report(report, 'json')
        if json_path.exists():
            quality_score += 5
            print(f"✅ JSON report saved: {json_path}")
        
        pdf_path = generator.save_report(report, 'pdf')
        if pdf_path.exists() or pdf_path.with_suffix('.txt').exists():
            quality_score += 5
            print(f"✅ PDF/Text report saved: {pdf_path}")
        
        # Calculate quality percentage
        quality_percentage = (quality_score / max_quality) * 100
        
        print(f"\n📊 Report Quality Score: {quality_score}/{max_quality} ({quality_percentage:.1f}%)")
        
        if quality_percentage >= 90:
            print("🎉 EXCELLENT: High-quality forensic reports!")
        elif quality_percentage >= 70:
            print("✅ GOOD: Quality forensic reports")
        elif quality_percentage >= 50:
            print("⚠️  FAIR: Report quality needs improvement")
        else:
            print("❌ POOR: Report quality is insufficient")
        
        return report, quality_percentage
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return None, 0

def test_workflow_integration():
    """Test end-to-end workflow integration"""
    
    print("\n🔄 Testing Workflow Integration")
    print("=" * 60)
    
    try:
        # Test workflow manager
        workflow = New_FORAI.ForensicWorkflowManager("FORENSIC-TEST", Path("D:/FORAI/test_output"), verbose=True)
        
        # Test chain of custody
        workflow.log_custody_event("TEST_START", "Beginning forensic accuracy test")
        workflow.log_custody_event("DATA_ANALYSIS", "Analyzing test evidence data")
        workflow.log_custody_event("REPORT_GENERATION", "Generating forensic reports")
        
        # Generate chain of custody report
        custody_file = workflow.generate_chain_of_custody_report()
        
        if custody_file.exists():
            print(f"✅ Chain of custody report generated: {custody_file}")
            
            # Check custody report content
            with open(custody_file, 'r') as f:
                custody_data = json.load(f)
            
            if len(custody_data.get('events', [])) >= 3:
                print(f"✅ Chain of custody contains {len(custody_data['events'])} events")
            else:
                print(f"⚠️  Chain of custody only contains {len(custody_data.get('events', []))} events")
        
        print("✅ Workflow integration test completed")
        return True
        
    except Exception as e:
        print(f"❌ Workflow integration failed: {e}")
        return False

def main():
    """Main forensic accuracy test function"""
    
    print("🚀 Comprehensive Forensic Accuracy Test for New_FORAI.py")
    print("Testing forensic question accuracy, report quality, and workflow integration")
    print("=" * 80)
    
    try:
        # Create comprehensive test data
        evidence_count = create_comprehensive_test_data()
        
        # Test forensic question accuracy
        question_results, question_accuracy = test_forensic_question_accuracy()
        
        # Test report generation quality
        report, report_quality = test_report_generation_quality()
        
        # Test workflow integration
        workflow_success = test_workflow_integration()
        
        # Calculate overall forensic readiness score
        overall_score = (question_accuracy + report_quality + (100 if workflow_success else 0)) / 3
        
        print(f"\n🎯 Overall Forensic Readiness Assessment")
        print("=" * 80)
        print(f"Evidence Records Created: {evidence_count}")
        print(f"Forensic Question Accuracy: {question_accuracy:.1f}%")
        print(f"Report Generation Quality: {report_quality:.1f}%")
        print(f"Workflow Integration: {'✅ PASS' if workflow_success else '❌ FAIL'}")
        print(f"Overall Forensic Score: {overall_score:.1f}%")
        
        if overall_score >= 85:
            print("\n🎉 FORENSIC READY: New_FORAI.py is ready for forensic investigations!")
            print("   - High accuracy in answering forensic questions")
            print("   - Quality report generation capabilities")
            print("   - Robust workflow integration")
        elif overall_score >= 70:
            print("\n✅ MOSTLY READY: New_FORAI.py shows good forensic capabilities")
            print("   - Some areas may need minor improvements")
        elif overall_score >= 50:
            print("\n⚠️  NEEDS IMPROVEMENT: New_FORAI.py requires enhancements")
            print("   - Several forensic capabilities need attention")
        else:
            print("\n❌ NOT READY: New_FORAI.py is not ready for forensic use")
            print("   - Significant improvements needed across all areas")
        
        # Save detailed results
        results_file = Path("D:/FORAI/forensic_accuracy_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        detailed_results = {
            'test_timestamp': datetime.now(timezone.utc).isoformat(),
            'evidence_count': evidence_count,
            'question_accuracy': question_accuracy,
            'report_quality': report_quality,
            'workflow_success': workflow_success,
            'overall_score': overall_score,
            'question_results': question_results,
            'forensic_questions': New_FORAI.FORENSIC_QUESTIONS
        }
        
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        print(f"\n📋 Detailed results saved to: {results_file}")
        
        return overall_score >= 70  # Return True if forensically ready
        
    except Exception as e:
        print(f"❌ Forensic accuracy test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)