#!/usr/bin/env python3
"""
XYZ.com Typosquatter Hunter - Specialized tool for finding *-xyz.com patterns

This tool specifically searches for typosquatting domains that follow the pattern
*-xyz.com where * is any 1-32 character alphanumeric combination, similar to
the malicious abc-xyz.com example.

Author: OpenHands
License: MIT
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
import string
import itertools


class XyzTyposquatterHunter:
    """Specialized hunter for *-xyz.com pattern typosquatters"""
    
    def __init__(self, delay: float = 1.0, timeout: int = 30):
        """
        Initialize the XYZ typosquatter hunter
        
        Args:
            delay: Delay between API requests to be respectful
            timeout: Request timeout in seconds
        """
        self.delay = delay
        self.timeout = timeout
        self.session_results = []

    def generate_xyz_variants(self, max_variants: int = 1000) -> Set[str]:
        """
        Generate potential *-xyz.com variants
        
        Args:
            max_variants: Maximum number of variants to generate
            
        Returns:
            Set of potential typosquatting domain variants following *-xyz.com pattern
        """
        variants = set()
        
        # Common prefixes that might be used for typosquatting
        common_prefixes = [
            # Geographic/location related
            'usa', 'uk', 'eu', 'asia', 'global', 'world', 'international',
            'north', 'south', 'east', 'west', 'central', 'pacific',
            'sydney', 'melbourne', 'brisbane', 'perth', 'adelaide',
            'nsw', 'vic', 'qld', 'wa', 'sa', 'nt', 'act', 'tas',
            
            # Business/organization related
            'bank', 'finance', 'pay', 'secure', 'safe', 'trust', 'official',
            'gov', 'govt', 'admin', 'portal', 'login', 'account', 'my',
            'customer', 'client', 'member', 'user', 'service', 'support',
            'help', 'info', 'news', 'update', 'alert', 'notice',
            
            # Common names that might be used
            'abc', 'john', 'mary', 'david', 'sarah', 'michael', 'emma',
            'james', 'anna', 'robert', 'kate', 'paul', 'jane', 'mark',
            
            # Technology/web related
            'web', 'online', 'digital', 'cyber', 'net', 'tech', 'app',
            'mobile', 'cloud', 'data', 'api', 'www', 'mail', 'ftp',
            
            # Action/verb related
            'get', 'buy', 'shop', 'pay', 'send', 'receive', 'transfer',
            'check', 'verify', 'confirm', 'update', 'renew', 'activate',
            
            # Descriptive words
            'new', 'old', 'real', 'true', 'fake', 'best', 'top', 'main',
            'primary', 'secondary', 'backup', 'temp', 'test', 'demo',
            
            # Numbers and combinations
            '1', '2', '3', '4', '5', '10', '20', '24', '365', '2024', '2025',
            
            # Single letters
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
        ]
        
        # Add all common prefixes
        for prefix in common_prefixes:
            if len(prefix) <= 32:
                variants.add(f"{prefix}-xyz.com")
        
        # Generate combinations of short prefixes
        short_prefixes = [p for p in common_prefixes if len(p) <= 4]
        for prefix1 in short_prefixes[:20]:  # Limit to prevent explosion
            for prefix2 in short_prefixes[:10]:
                combined = f"{prefix1}{prefix2}"
                if len(combined) <= 32:
                    variants.add(f"{combined}-xyz.com")
        
        # Generate numeric patterns
        for i in range(1, 1000):  # 1-999
            num_str = str(i)
            if len(num_str) <= 32:
                variants.add(f"{num_str}-xyz.com")
        
        # Generate letter combinations (2-4 characters)
        letters = string.ascii_lowercase
        for length in range(2, 5):  # 2, 3, 4 character combinations
            count = 0
            for combo in itertools.product(letters, repeat=length):
                if count >= 100:  # Limit to prevent too many combinations
                    break
                prefix = ''.join(combo)
                variants.add(f"{prefix}-xyz.com")
                count += 1
        
        # Generate alphanumeric combinations (common patterns)
        alphanumeric_patterns = []
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            for num in range(1, 100):
                alphanumeric_patterns.extend([
                    f"{letter}{num}",
                    f"{num}{letter}",
                    f"{letter}{letter}{num}",
                    f"{num}{letter}{letter}"
                ])
        
        for pattern in alphanumeric_patterns[:500]:  # Limit to first 500
            if len(pattern) <= 32:
                variants.add(f"{pattern}-xyz.com")
        
        # Limit total variants
        if len(variants) > max_variants:
            variants = set(list(variants)[:max_variants])
        
        return variants

    def query_crtsh(self, domain: str) -> List[Dict]:
        """
        Query crt.sh for certificates matching the domain
        
        Args:
            domain: Domain to search for
            
        Returns:
            List of certificate records from crt.sh
        """
        try:
            # URL encode the domain and construct the query
            encoded_domain = quote(domain)
            url = f"https://crt.sh/?q={encoded_domain}&output=json"
            
            # Create request with user agent
            req = Request(url)
            req.add_header('User-Agent', 'XyzTyposquatterHunter/1.0')
            
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
                print(f"HTTP Error {e.code} for domain {domain}: {e.reason}")
                return []
        except URLError as e:
            print(f"URL Error for domain {domain}: {e.reason}")
            return []
        except json.JSONDecodeError:
            print(f"Invalid JSON response for domain {domain}")
            return []
        except Exception as e:
            print(f"Unexpected error for domain {domain}: {e}")
            return []

    def analyze_certificates(self, domain: str, certificates: List[Dict]) -> Dict:
        """
        Analyze certificates for suspicious patterns
        
        Args:
            domain: The domain being analyzed
            certificates: List of certificate records
            
        Returns:
            Analysis results dictionary
        """
        if not certificates:
            return {
                'domain': domain,
                'certificate_count': 0,
                'certificates': [],
                'risk_score': 0,
                'risk_factors': []
            }
        
        risk_factors = []
        risk_score = 0
        
        # Analyze certificate patterns
        issuers = defaultdict(int)
        common_names = set()
        
        for cert in certificates:
            issuer = cert.get('issuer_name', 'Unknown')
            issuers[issuer] += 1
            
            common_name = cert.get('common_name', '')
            if common_name:
                common_names.add(common_name)
        
        # Risk factor: Multiple different issuers (could indicate suspicious activity)
        if len(issuers) > 3:
            risk_factors.append(f"Multiple certificate issuers ({len(issuers)})")
            risk_score += 3
        
        # Risk factor: High number of certificates
        if len(certificates) > 10:
            risk_factors.append(f"High number of certificates ({len(certificates)})")
            risk_score += 2
        elif len(certificates) > 5:
            risk_factors.append(f"Moderate number of certificates ({len(certificates)})")
            risk_score += 1
        
        # Risk factor: Recent certificate activity
        recent_certs = 0
        current_year = datetime.now().year
        for cert in certificates:
            not_before = cert.get('not_before', '')
            if not_before and str(current_year) in not_before:
                recent_certs += 1
        
        if recent_certs > 0:
            risk_factors.append(f"Recent certificate activity ({recent_certs} this year)")
            risk_score += 2
        
        # Additional risk factor: Suspicious prefix patterns
        prefix = domain.split('-xyz.com')[0] if '-xyz.com' in domain else ''
        suspicious_prefixes = ['abc', 'secure', 'official', 'bank', 'pay', 'login', 'account']
        if prefix.lower() in suspicious_prefixes:
            risk_factors.append(f"Suspicious prefix pattern: '{prefix}'")
            risk_score += 3
        
        return {
            'domain': domain,
            'certificate_count': len(certificates),
            'certificates': certificates,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'issuers': dict(issuers),
            'common_names': list(common_names),
            'prefix': prefix
        }

    def hunt_xyz_typosquatters(self, max_variants: int = 1000, min_risk_score: int = 0) -> List[Dict]:
        """
        Hunt for *-xyz.com typosquatters
        
        Args:
            max_variants: Maximum number of variants to check
            min_risk_score: Minimum risk score to include in results
            
        Returns:
            List of analysis results for suspicious domains
        """
        print("Generating *-xyz.com typosquatting variants...")
        variants = self.generate_xyz_variants(max_variants)
        
        print(f"Generated {len(variants)} variants to check")
        print("Querying crt.sh for certificates...")
        print("This may take a while - please be patient...")
        print()
        
        results = []
        total_variants = len(variants)
        found_count = 0
        
        for i, variant in enumerate(variants, 1):
            if i % 50 == 0 or i == total_variants:
                print(f"Progress: {i}/{total_variants} ({i/total_variants*100:.1f}%) - Found: {found_count}")
            
            certificates = self.query_crtsh(variant)
            if certificates:
                analysis = self.analyze_certificates(variant, certificates)
                if analysis['risk_score'] >= min_risk_score:
                    results.append(analysis)
                    found_count += 1
                    prefix = analysis['prefix']
                    print(f"  ⚠️  FOUND: {variant} (prefix: '{prefix}', risk: {analysis['risk_score']}, certs: {analysis['certificate_count']})")
        
        # Sort by risk score (highest first)
        results.sort(key=lambda x: x['risk_score'], reverse=True)
        
        self.session_results = results
        return results

    def export_results(self, results: List[Dict], output_file: str, format_type: str = 'json'):
        """
        Export results to file
        
        Args:
            results: Analysis results to export
            output_file: Output file path
            format_type: Export format ('json', 'csv', 'txt')
        """
        if format_type.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        elif format_type.lower() == 'csv':
            import csv
            with open(output_file, 'w', newline='') as f:
                if not results:
                    return
                
                writer = csv.writer(f)
                writer.writerow(['Domain', 'Prefix', 'Certificate Count', 'Risk Score', 'Risk Factors', 'Issuers'])
                
                for result in results:
                    writer.writerow([
                        result['domain'],
                        result['prefix'],
                        result['certificate_count'],
                        result['risk_score'],
                        '; '.join(result['risk_factors']),
                        '; '.join(result['issuers'].keys())
                    ])
        
        elif format_type.lower() == 'txt':
            with open(output_file, 'w') as f:
                f.write(f"XYZ.com Typosquatter Hunt Results\n")
                f.write(f"Pattern: *-xyz.com\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total suspicious domains found: {len(results)}\n\n")
                
                for result in results:
                    f.write(f"Domain: {result['domain']}\n")
                    f.write(f"Prefix: {result['prefix']}\n")
                    f.write(f"Risk Score: {result['risk_score']}\n")
                    f.write(f"Certificate Count: {result['certificate_count']}\n")
                    f.write(f"Risk Factors: {', '.join(result['risk_factors'])}\n")
                    f.write(f"Issuers: {', '.join(result['issuers'].keys())}\n")
                    f.write("-" * 60 + "\n")
        
        print(f"Results exported to: {output_file}")


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Hunt for *-xyz.com typosquatting domains using Certificate Transparency logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool specifically searches for domains following the pattern *-xyz.com where * is any
1-32 character alphanumeric combination, similar to the malicious abc-xyz.com example.

Examples:
  %(prog)s
  %(prog)s --max-variants 2000 --min-risk-score 2
  %(prog)s --output xyz_typosquatters.json --format json
  %(prog)s --delay 2.0 --timeout 60 --verbose
        """
    )
    
    parser.add_argument('--max-variants', type=int, default=1000,
                       help='Maximum number of domain variants to generate (default: 1000)')
    parser.add_argument('--min-risk-score', type=int, default=0,
                       help='Minimum risk score to include in results (default: 0)')
    parser.add_argument('--delay', type=float, default=1.0,
                       help='Delay between API requests in seconds (default: 1.0)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['json', 'csv', 'txt'], default='json',
                       help='Output format (default: json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Initialize hunter
    hunter = XyzTyposquatterHunter(delay=args.delay, timeout=args.timeout)
    
    try:
        # Hunt for typosquatters
        results = hunter.hunt_xyz_typosquatters(
            max_variants=args.max_variants,
            min_risk_score=args.min_risk_score
        )
        
        # Display results
        print(f"\n{'='*80}")
        print(f"XYZ.COM TYPOSQUATTER HUNT RESULTS")
        print(f"{'='*80}")
        print(f"Pattern searched: *-xyz.com")
        print(f"Suspicious domains found: {len(results)}")
        
        if results:
            print(f"\nSuspicious *-xyz.com domains found:")
            print(f"{'Rank':<4} {'Domain':<25} {'Prefix':<15} {'Risk':<4} {'Certs':<5} {'Risk Factors'}")
            print("-" * 80)
            
            for i, result in enumerate(results, 1):
                risk_factors_str = ', '.join(result['risk_factors'][:2])  # Show first 2 factors
                if len(result['risk_factors']) > 2:
                    risk_factors_str += "..."
                
                print(f"{i:<4} {result['domain']:<25} {result['prefix']:<15} "
                      f"{result['risk_score']:<4} {result['certificate_count']:<5} {risk_factors_str}")
                
                if args.verbose and result['risk_factors']:
                    for factor in result['risk_factors']:
                        print(f"     └─ {factor}")
        else:
            print("No suspicious *-xyz.com domains found.")
        
        # Export results if requested
        if args.output:
            hunter.export_results(results, args.output, args.format)
        
        print(f"\n{'='*80}")
        print("IMPORTANT NOTES:")
        print("• Always manually verify suspicious domains before taking action")
        print("• This tool only detects domains with SSL/TLS certificates in CT logs")
        print("• Consider legal implications before reporting suspected typosquatters")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()