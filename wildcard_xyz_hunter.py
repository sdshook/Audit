#!/usr/bin/env python3
"""
Wildcard XYZ.com Typosquatter Hunter - Efficient wildcard-based search

This tool uses crt.sh's wildcard support to efficiently find potential
typosquatting domains related to xyz.com, including the *-xyz.com pattern
like the malicious abc-xyz.com example.

Author: Shane D. Shook (C) All Rights Reserved
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Set
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class WildcardXyzHunter:
    """Efficient wildcard-based hunter for XYZ.com typosquatters"""
    
    def __init__(self, delay: float = 2.0, timeout: int = 30):
        """
        Initialize the wildcard XYZ hunter
        
        Args:
            delay: Delay between API requests to be respectful
            timeout: Request timeout in seconds
        """
        self.delay = delay
        self.timeout = timeout
        self.session_results = []

    def query_crtsh_wildcard(self, pattern: str) -> List[Dict]:
        """
        Query crt.sh using wildcard patterns
        
        Args:
            pattern: Wildcard pattern to search for (e.g., "%.xyz.com")
            
        Returns:
            List of certificate records from crt.sh
        """
        try:
            # URL encode the pattern
            encoded_pattern = quote(pattern)
            url = f"https://crt.sh/?q={encoded_pattern}&output=json"
            
            print(f"Querying: {pattern}")
            
            # Create request with user agent
            req = Request(url)
            req.add_header('User-Agent', 'WildcardXyzHunter/1.0')
            
            # Make the request
            with urlopen(req, timeout=self.timeout) as response:
                data = response.read().decode('utf-8')
                
            # Parse JSON response
            certificates = json.loads(data)
            
            # Add delay to be respectful to the API
            time.sleep(self.delay)
            
            return certificates if certificates else []
            
        except HTTPError as e:
            if e.code == 404:
                return []  # No certificates found
            else:
                print(f"HTTP Error {e.code} for pattern {pattern}: {e.reason}")
                return []
        except URLError as e:
            print(f"URL Error for pattern {pattern}: {e.reason}")
            return []
        except json.JSONDecodeError as e:
            print(f"Invalid JSON response for pattern {pattern}: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error for pattern {pattern}: {e}")
            return []

    def extract_domains_from_certificates(self, certificates: List[Dict]) -> Set[str]:
        """
        Extract unique domain names from certificate records
        
        Args:
            certificates: List of certificate records
            
        Returns:
            Set of unique domain names
        """
        domains = set()
        
        for cert in certificates:
            # Get common name
            common_name = cert.get('common_name', '')
            if common_name:
                domains.add(common_name.lower())
            
            # Get subject alternative names
            name_value = cert.get('name_value', '')
            if name_value:
                # Split by newlines and add each domain
                for domain in name_value.split('\n'):
                    domain = domain.strip().lower()
                    if domain:
                        domains.add(domain)
        
        return domains

    def filter_xyz_related_domains(self, domains: Set[str]) -> Dict[str, List[str]]:
        """
        Filter and categorize XYZ-related domains
        
        Args:
            domains: Set of domain names to filter
            
        Returns:
            Dictionary categorizing different types of XYZ-related domains
        """
        categorized = {
            'hyphen_xyz_com': [],      # *-xyz.com pattern
            'subdomain_xyz_com': [],   # *.xyz.com pattern  
            'xyz_variations': [],      # variations of xyz.com
            'suspicious_patterns': []  # other suspicious patterns
        }
        
        for domain in domains:
            # Skip wildcard domains (those starting with *)
            if domain.startswith('*'):
                continue
                
            # Pattern: *-xyz.com (like abc-xyz.com)
            if re.match(r'^[a-zA-Z0-9]+-xyz\.com$', domain):
                categorized['hyphen_xyz_com'].append(domain)
            
            # Pattern: *.xyz.com (subdomains of xyz.com)
            elif domain.endswith('.xyz.com') and domain != 'xyz.com':
                categorized['subdomain_xyz_com'].append(domain)
            
            # Variations of xyz.com (typos, similar domains)
            elif 'xyz' in domain and domain.endswith('.com'):
                # Check for common typosquatting patterns
                if any(pattern in domain for pattern in ['xyz', 'xzy', 'yxz', 'zyx', 'xyc']):
                    if domain not in ['xyz.com']:  # Exclude the legitimate domain
                        categorized['xyz_variations'].append(domain)
            
            # Other suspicious patterns containing 'xyz'
            elif 'xyz' in domain:
                categorized['suspicious_patterns'].append(domain)
        
        # Sort each category
        for category in categorized:
            categorized[category].sort()
        
        return categorized

    def analyze_domain_risk(self, domain: str, certificates: List[Dict]) -> Dict:
        """
        Analyze risk level for a specific domain
        
        Args:
            domain: Domain to analyze
            certificates: All certificates containing this domain
            
        Returns:
            Risk analysis dictionary
        """
        # Filter certificates that contain this specific domain
        domain_certs = []
        for cert in certificates:
            common_name = cert.get('common_name', '').lower()
            name_value = cert.get('name_value', '').lower()
            
            if domain in common_name or domain in name_value:
                domain_certs.append(cert)
        
        risk_score = 0
        risk_factors = []
        
        # Analyze certificate patterns
        issuers = defaultdict(int)
        for cert in domain_certs:
            issuer = cert.get('issuer_name', 'Unknown')
            issuers[issuer] += 1
        
        # Risk factor: Multiple different issuers
        if len(issuers) > 2:
            risk_factors.append(f"Multiple certificate issuers ({len(issuers)})")
            risk_score += 3
        
        # Risk factor: High number of certificates
        if len(domain_certs) > 5:
            risk_factors.append(f"High number of certificates ({len(domain_certs)})")
            risk_score += 2
        elif len(domain_certs) > 2:
            risk_factors.append(f"Multiple certificates ({len(domain_certs)})")
            risk_score += 1
        
        # Risk factor: Recent certificate activity
        recent_certs = 0
        current_year = datetime.now().year
        for cert in domain_certs:
            not_before = cert.get('not_before', '')
            if not_before and str(current_year) in not_before:
                recent_certs += 1
        
        if recent_certs > 0:
            risk_factors.append(f"Recent certificate activity ({recent_certs} this year)")
            risk_score += 2
        
        # Risk factor: Suspicious domain patterns
        if '-xyz.com' in domain:
            prefix = domain.split('-xyz.com')[0]
            suspicious_prefixes = [
                'abc', 'secure', 'official', 'bank', 'pay', 'login', 'account',
                'admin', 'portal', 'customer', 'support', 'help', 'service'
            ]
            if prefix in suspicious_prefixes:
                risk_factors.append(f"High-risk prefix: '{prefix}'")
                risk_score += 4
            else:
                risk_factors.append(f"Hyphen pattern with prefix: '{prefix}'")
                risk_score += 2
        
        return {
            'domain': domain,
            'certificate_count': len(domain_certs),
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'issuers': dict(issuers),
            'certificates': domain_certs
        }

    def hunt_wildcard_xyz_typosquatters(self) -> Dict:
        """
        Hunt for XYZ.com typosquatters using wildcard searches
        
        Returns:
            Dictionary containing categorized results
        """
        print("=" * 80)
        print("WILDCARD XYZ.COM TYPOSQUATTER HUNTER")
        print("=" * 80)
        print("Using efficient wildcard searches to find XYZ-related domains...")
        print()
        
        all_certificates = []
        all_domains = set()
        
        # Search patterns to use
        search_patterns = [
            "%.xyz.com",      # All subdomains of xyz.com
            # Note: %-xyz.com doesn't work, so we'll search for common prefixes manually
        ]
        
        # Add common prefix patterns for *-xyz.com
        common_prefixes = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'abc', 'bank', 'secure', 'pay', 'admin', 'login', 'official',
            'customer', 'support', 'help', 'service', 'portal', 'account'
        ]
        
        # Add specific *-xyz.com patterns
        for prefix in common_prefixes:
            search_patterns.append(f"{prefix}-xyz.com")
        
        print(f"Searching {len(search_patterns)} patterns...")
        
        # Execute searches
        for i, pattern in enumerate(search_patterns, 1):
            print(f"[{i}/{len(search_patterns)}] Searching: {pattern}")
            
            certificates = self.query_crtsh_wildcard(pattern)
            if certificates:
                all_certificates.extend(certificates)
                domains = self.extract_domains_from_certificates(certificates)
                all_domains.update(domains)
                print(f"  Found {len(certificates)} certificates, {len(domains)} unique domains")
            else:
                print(f"  No results")
        
        print(f"\nTotal unique domains found: {len(all_domains)}")
        print("Categorizing and analyzing domains...")
        
        # Categorize domains
        categorized_domains = self.filter_xyz_related_domains(all_domains)
        
        # Analyze risk for each domain
        results = {}
        for category, domains in categorized_domains.items():
            if not domains:
                continue
                
            print(f"\nAnalyzing {len(domains)} domains in category: {category}")
            category_results = []
            
            for domain in domains:
                analysis = self.analyze_domain_risk(domain, all_certificates)
                category_results.append(analysis)
            
            # Sort by risk score (highest first)
            category_results.sort(key=lambda x: x['risk_score'], reverse=True)
            results[category] = category_results
        
        return results

    def display_results(self, results: Dict):
        """
        Display the hunt results in a formatted way
        
        Args:
            results: Categorized results dictionary
        """
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        
        total_domains = sum(len(category_results) for category_results in results.values())
        high_risk_domains = sum(
            len([d for d in category_results if d['risk_score'] >= 4])
            for category_results in results.values()
        )
        
        print(f"Total suspicious domains found: {total_domains}")
        print(f"High-risk domains (score ≥ 4): {high_risk_domains}")
        print()
        
        # Display each category
        category_names = {
            'hyphen_xyz_com': 'HYPHEN PATTERN (*-xyz.com)',
            'subdomain_xyz_com': 'SUBDOMAIN PATTERN (*.xyz.com)',
            'xyz_variations': 'XYZ VARIATIONS',
            'suspicious_patterns': 'OTHER SUSPICIOUS PATTERNS'
        }
        
        for category, category_results in results.items():
            if not category_results:
                continue
                
            print(f"\n{category_names.get(category, category.upper())}")
            print("-" * 60)
            
            if category == 'hyphen_xyz_com':
                print("⚠️  These domains follow the *-xyz.com pattern like abc-xyz.com")
                print()
            
            for i, domain_analysis in enumerate(category_results[:20], 1):  # Show top 20
                domain = domain_analysis['domain']
                risk_score = domain_analysis['risk_score']
                cert_count = domain_analysis['certificate_count']
                
                risk_indicator = "🔴" if risk_score >= 4 else "🟡" if risk_score >= 2 else "🟢"
                
                print(f"{i:2d}. {risk_indicator} {domain:<30} (risk: {risk_score}, certs: {cert_count})")
                
                # Show risk factors for high-risk domains
                if risk_score >= 4 and domain_analysis['risk_factors']:
                    for factor in domain_analysis['risk_factors'][:2]:  # Show first 2 factors
                        print(f"     └─ {factor}")
            
            if len(category_results) > 20:
                print(f"     ... and {len(category_results) - 20} more domains")

    def export_results(self, results: Dict, output_file: str, format_type: str = 'json'):
        """
        Export results to file
        
        Args:
            results: Results dictionary to export
            output_file: Output file path
            format_type: Export format ('json', 'csv', 'txt')
        """
        if format_type.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format_type.lower() == 'csv':
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Category', 'Domain', 'Risk Score', 'Certificate Count', 'Risk Factors', 'Issuers'])
                
                for category, category_results in results.items():
                    for result in category_results:
                        writer.writerow([
                            category,
                            result['domain'],
                            result['risk_score'],
                            result['certificate_count'],
                            '; '.join(result['risk_factors']),
                            '; '.join(result['issuers'].keys())
                        ])
        
        elif format_type.lower() == 'txt':
            with open(output_file, 'w') as f:
                f.write(f"Wildcard XYZ.com Typosquatter Hunt Results\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"=" * 60 + "\n\n")
                
                for category, category_results in results.items():
                    if not category_results:
                        continue
                        
                    f.write(f"{category.upper().replace('_', ' ')}\n")
                    f.write("-" * 40 + "\n")
                    
                    for result in category_results:
                        f.write(f"Domain: {result['domain']}\n")
                        f.write(f"Risk Score: {result['risk_score']}\n")
                        f.write(f"Certificate Count: {result['certificate_count']}\n")
                        f.write(f"Risk Factors: {', '.join(result['risk_factors'])}\n")
                        f.write(f"Issuers: {', '.join(result['issuers'].keys())}\n")
                        f.write("-" * 40 + "\n")
                    f.write("\n")
        
        print(f"Results exported to: {output_file}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Hunt for XYZ.com typosquatters using efficient wildcard searches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool uses crt.sh's wildcard support to efficiently find potential
typosquatting domains related to xyz.com, including the *-xyz.com pattern
like the malicious abc-xyz.com example.

The tool searches for:
- *.xyz.com (subdomains)
- *-xyz.com patterns (like abc-xyz.com)
- Other XYZ-related variations

Examples:
  %(prog)s
  %(prog)s --output xyz_results.json --format json
  %(prog)s --delay 3.0 --timeout 60
        """
    )
    
    parser.add_argument('--delay', type=float, default=2.0,
                       help='Delay between API requests in seconds (default: 2.0)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv', 'txt'], default='json',
                       help='Output format (default: json)')
    
    args = parser.parse_args()
    
    # Initialize hunter
    hunter = WildcardXyzHunter(delay=args.delay, timeout=args.timeout)
    
    try:
        # Hunt for typosquatters
        results = hunter.hunt_wildcard_xyz_typosquatters()
        
        # Display results
        hunter.display_results(results)
        
        # Export results if requested
        if args.output:
            hunter.export_results(results, args.output, args.format)
        
        print("\n" + "=" * 80)
        print("IMPORTANT NOTES:")
        print("• Always manually verify suspicious domains before taking action")
        print("• This tool only detects domains with SSL/TLS certificates in CT logs")
        print("• The *-xyz.com pattern search covers common prefixes")
        print("• Consider legal implications before reporting suspected typosquatters")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
