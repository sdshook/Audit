<<<<<<< HEAD
# SIPCompare v2.0: Advanced Forensic Code Similarity Analysis Tool

## Overview

SIPCompare (Software Intellectual Property Compare) is a comprehensive forensic tool designed to detect software intellectual property theft and code plagiarism with advanced obfuscation resistance, statistical analysis, and forensic-quality reporting.

## Key Features

### 🔍 **Advanced Detection Capabilities**
- **Multi-dimensional Analysis**: Combines token-based, semantic, structural, and control-flow similarity
- **Obfuscation Resistance**: Detects similarities despite variable renaming, code reordering, and style changes
- **Clone Type Classification**: Industry-standard Type 1-4 clone detection
- **Cross-Language Support**: Supports 15+ programming languages including Python, Java, C/C++, JavaScript, Go, Rust, and more

### 📊 **Statistical Rigor**
- **Statistical Significance Testing**: P-values and confidence intervals for all matches
- **Baseline Establishment**: Automatic baseline calculation from random code samples
- **Evidence Strength Classification**: STRONG/MODERATE/WEAK evidence categories
- **Obfuscation Detection**: Identifies deliberate code modification attempts

### 🚀 **Performance & Scalability**
- **Parallel Processing**: Multi-core support for large repositories
- **Memory Efficient**: Optimized for large codebases
- **Advanced Embeddings**: GraphCodeBERT, CodeT5, and MiniLM support
- **AST-Based Analysis**: Tree-sitter parsers for accurate structural analysis

### 📋 **Forensic-Quality Reporting**
- **Comprehensive Evidence Packages**: ZIP archives with all analysis data
- **Multiple Report Formats**: HTML, CSV, JSON, and executive summaries
- **Chain of Custody**: Forensic documentation for legal proceedings
- **Code Diffs**: Side-by-side comparisons with transformation detection

## Installation

### Prerequisites
```bash
pip install numpy scipy tqdm
pip install sentence-transformers  # For MiniLM embeddings
pip install transformers torch     # For GraphCodeBERT/CodeT5
pip install tree-sitter           # For AST analysis (optional)
```

### Quick Start
```bash
git clone <repository-url>
cd SIPCompare
python SIPCompare.py --repoA /path/to/repo1 --repoB /path/to/repo2
```

## Usage

### Basic Analysis
```bash
python SIPCompare.py --repoA /path/to/repo1 --repoB /path/to/repo2
```

### High-Sensitivity Analysis
```bash
python SIPCompare.py --repoA /path/to/repo1 --repoB /path/to/repo2 \
                     --threshold 0.6 --parallel 4 --embedding-model graphcodebert
```

### Cross-Language Detection
```bash
python SIPCompare.py --repoA /path/to/python_repo --repoB /path/to/java_repo \
                     --cross-language --embedding-model codet5
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--repoA` | Path to first repository | Required |
| `--repoB` | Path to second repository | Required |
| `--threshold` | Similarity threshold (0-1) | 0.75 |
| `--embedding-model` | Model: mini/graphcodebert/codet5 | graphcodebert |
| `--parallel` | Number of parallel processes | 1 |
| `--no-statistical` | Disable statistical testing | False |
| `--cross-language` | Enable cross-language detection | False |
| `--output` | Output evidence package | evidence_package.zip |
| `--verbose` | Enable verbose logging | False |

## Supported Languages

| Language | Extension | AST Support | Semantic Analysis |
|----------|-----------|-------------|-------------------|
| Python | .py | ✅ | ✅ |
| Java | .java | ✅ | ✅ |
| C/C++ | .c, .cpp, .h, .hpp | ✅ | ✅ |
| JavaScript | .js | ✅ | ✅ |
| TypeScript | .ts | ✅ | ✅ |
| Go | .go | ✅ | ✅ |
| Rust | .rs | ✅ | ✅ |
| C# | .cs | ✅ | ✅ |
| PHP | .php | ✅ | ✅ |
| Ruby | .rb | ✅ | ✅ |
| Swift | .swift | ✅ | ✅ |
| Kotlin | .kt | ✅ | ✅ |
| Scala | .scala | ✅ | ✅ |
| Shell Scripts | .sh, .zsh, .ksh | ❌ | ✅ |
| PowerShell | .ps1 | ❌ | ✅ |

## Clone Type Classification

SIPCompare implements industry-standard clone detection:

### Type 1: Exact Clones
- Identical code except for whitespace and comments
- **Detection**: Hash-based comparison after normalization
- **Evidence**: STRONG (similarity > 0.95)

### Type 2: Renamed Clones  
- Identical structure with renamed identifiers
- **Detection**: AST structural comparison + identifier normalization
- **Evidence**: STRONG (similarity > 0.85, statistically significant)

### Type 3: Near-Miss Clones
- Similar code with statement additions/deletions
- **Detection**: Longest Common Subsequence + structural analysis
- **Evidence**: MODERATE (similarity > 0.75, statistically significant)

### Type 4: Semantic Clones
- Different syntax, same functionality
- **Detection**: Semantic embeddings + complexity analysis
- **Evidence**: MODERATE (similarity > 0.65, statistically significant)

## Output Analysis

### Evidence Package Structure
```
evidence_package.zip
├── reports/
│   ├── forensic_report.html      # Interactive HTML report
│   ├── detailed_analysis.csv     # Machine-readable data
│   ├── analysis_data.json        # Complete analysis data
│   ├── executive_summary.txt     # Non-technical summary
│   └── technical_analysis.txt    # Detailed technical report
├── evidence_files/               # Source code files
│   ├── repo_a/
│   └── repo_b/
└── chain_of_custody.txt         # Forensic documentation
```

### Interpreting Results

#### Evidence Strength
- **STRONG**: High confidence of IP theft (legal action recommended)
- **MODERATE**: Significant similarity (further investigation needed)
- **WEAK**: Low similarity (likely coincidental)

#### Statistical Significance
- **p < 0.05**: Statistically significant match
- **Confidence Intervals**: Range of similarity uncertainty
- **Baseline Comparison**: Comparison against random code samples

#### Obfuscation Detection
- **Identifier Renaming**: Systematic variable/function renaming
- **Statement Modification**: Code additions/deletions
- **Control Flow Changes**: Altered program structure
- **Multiple Patterns**: Strong indicator of deliberate obfuscation

## Advanced Features

### Code Normalization
- **Identifier Canonicalization**: Maps variables to canonical names
- **Construct Normalization**: Handles equivalent language constructs
- **Comment/String Removal**: Focuses on functional code
- **Import Sorting**: Normalizes dependency declarations

### Structural Analysis
- **AST Depth Calculation**: Measures code complexity
- **Control Flow Extraction**: Identifies program flow patterns
- **Function Signature Analysis**: Compares method structures
- **Complexity Metrics**: Cyclomatic complexity calculation

### Semantic Analysis
- **Transformer Models**: State-of-the-art code embeddings
- **Contextual Understanding**: Captures semantic meaning
- **Cross-Language Capability**: Language-agnostic analysis
- **Chunked Processing**: Handles large files efficiently

## Performance Considerations

### Memory Usage
- **Embedding Caching**: Reuses computed embeddings
- **Streaming Processing**: Processes files incrementally
- **Garbage Collection**: Automatic memory management

### Processing Speed
- **Parallel Processing**: Utilizes multiple CPU cores
- **Early Termination**: Skips low-similarity comparisons
- **Optimized Algorithms**: Efficient similarity calculations

### Scalability
- **Large Repositories**: Tested on 10,000+ file repositories
- **Memory Efficient**: Handles GB-sized codebases
- **Progress Tracking**: Real-time progress indicators

## Legal and Ethical Considerations

### Forensic Integrity
- **Chain of Custody**: Complete audit trail
- **Hash Verification**: File integrity validation
- **Reproducible Results**: Deterministic analysis
- **Expert Testimony**: Court-admissible evidence

### Privacy and Security
- **Local Processing**: No data transmitted externally
- **Secure Storage**: Encrypted evidence packages
- **Access Control**: Restricted file permissions
- **Data Retention**: Configurable cleanup policies

## Troubleshooting

### Common Issues

#### "No processable files found"
- Check file extensions are supported
- Verify repository paths are correct
- Ensure files are readable

#### "Model loading failed"
- Install required dependencies
- Check internet connection for model downloads
- Verify sufficient disk space

#### "Out of memory"
- Reduce parallel workers
- Use 'mini' embedding model
- Process smaller file sets

#### "Tree-sitter not available"
- Install tree-sitter: `pip install tree-sitter`
- Tool falls back to regex-based analysis
- Reduced accuracy but still functional

### Performance Tuning

#### For Large Repositories
```bash
python SIPCompare.py --repoA large_repo1 --repoB large_repo2 \
                     --parallel 8 --embedding-model mini \
                     --threshold 0.8
```

#### For High Accuracy
```bash
python SIPCompare.py --repoA repo1 --repoB repo2 \
                     --threshold 0.6 --embedding-model graphcodebert \
                     --verbose
```

#### For Cross-Language Analysis
```bash
python SIPCompare.py --repoA python_repo --repoB java_repo \
                     --embedding-model codet5 --cross-language \
                     --threshold 0.7
```

## Contributing

### Development Setup
```bash
git clone <repository-url>
cd SIPCompare
pip install -r requirements.txt
python -m pytest tests/
```

### Code Quality
- **Type Hints**: Full type annotation
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit and integration tests
- **Linting**: PEP 8 compliance

### Feature Requests
- Open GitHub issues for new features
- Include use cases and examples
- Consider performance implications
- Maintain forensic integrity

## License

This software is proprietary and confidential. Unauthorized use, distribution, or modification is strictly prohibited.

© 2025 Shane D. Shook, All Rights Reserved

## Support

For technical support, feature requests, or licensing inquiries, please contact:
- Email: [support email]
- Documentation: [documentation URL]
- Issues: [GitHub issues URL]

---

**Disclaimer**: This tool is designed for legitimate intellectual property protection and forensic analysis. Users are responsible for ensuring compliance with applicable laws and regulations. The authors assume no liability for misuse of this software.
=======
# Security Audit Toolkit

A comprehensive collection of security auditing and forensic analysis tools developed by Shane Shook (c) 2025.

## Overview

This repository contains various PowerShell and Python scripts designed to automate security assessments, incident response, and forensic analysis across different platforms and cloud services.

## Tool Categories

### Cloud Services Reports (CSR)
Scripts to automate collection of cloud services activities for security posture assessment and incident triage:

* **[AzUAR.ps1](AzUAR.ps1)** - PowerShell script to pull Azure Active Directory authentication activities by user(s)  
  *Requires PowerShell 7 and .NET 8*

* **[o365UAR.ps1](o365UAR.ps1)** - PowerShell script to pull Unified Audit Logs from Office 365 by user(s)

* **[o365AppsChanges.ps1](o365AppsChanges.ps1)** - PowerShell script to pull changes made to applications by user(s)

* **[GUAR.py](GUAR.py)** - Python script to pull Unified Activity Logs from Google Workspace by user(s)

### Cyber Risk Assessments (CRA)
Scripts to automate collection of security posture information:

* **[CRA_Win.ps1](CRA_Win.ps1)** - Windows assessment (run on each host)
* **[CRA_LM.sh](CRA_LM.sh)** - Linux & Mac assessment (run on each host)  
* **[CRA_AD.ps1](CRA_AD.ps1)** - Active Directory assessment (for on-premise AD, run only once from any domain-connected host)

### Cyber Breach Assessment (CBA)
Scripts to automate collection of security posture information for incident triage:

* **[CBA_Win.ps1](CBA_Win.ps1)** - Cyber Breach Assessment script for Windows systems
* **[CBT.ps1](CBT.ps1)** - Cyber Breach Triage script to quickly collect active communicating processes for incident triage

### Forensic Analysis Tools
Advanced forensic analysis and AI-assisted investigation tools:

* **[FORAI.py](FORAI.py)** - AI-assisted forensic analysis tool for Windows systems
* **[FORAIR.py](FORAIR.py)** - Related forensic analysis tool
* **[FORAI Workflow Documentation](FORAI_Workflow.md)** - Detailed workflow and usage guide for FORAI.py

### Additional Security Tools

* **[UserRisk.ps1](UserRisk.ps1)** - User risk assessment script
* **[UserRisk_JITA.ps1](UserRisk_JITA.ps1)** - Just-in-time access user risk assessment
* **[PermissionRisk.ps1](PermissionRisk.ps1)** - Permission risk analysis script
* **[AzRoleChanges.ps1](AzRoleChanges.ps1)** - Azure role changes monitoring script
* **[AppSecretsChanges_JITA.ps1](AppSecretsChanges_JITA.ps1)** - Application secrets changes monitoring (JITA)
* **[AppSecretsChanges_persistent.ps1](AppSecretsChanges_persistent.ps1)** - Application secrets changes monitoring (persistent)
* **[GBCRA_Win.ps1](GBCRA_Win.ps1)** - Group-based cyber risk assessment for Windows
* **[AdminCreds/](AdminCreds/)** - Administrative credentials management tools

## Acknowledgments

Special thanks to Brandon Pimentel for his valuable contributions to this project.
>>>>>>> 0c1a399e5690e0f7c2bbf986e59146f823d5c8cb
