# SIPCompare v2.0: Forensic Code Similarity Analysis Tool

## Overview

SIPCompare (Software Intellectual Property Compare) detects software intellectual property theft and code plagiarism using AI models, multi-dimensional analysis, and statistical validation to provide court-admissible evidence across 15+ programming languages.

## Key Features

- **Multi-dimensional Analysis**: Token-based, semantic, structural, and control-flow similarity detection
- **Obfuscation Resistance**: Detects similarities despite variable renaming, reordering, and style changes  
- **Cross-Language Support**: Python↔Java, C++↔Python, and other language pairs
- **Statistical Validation**: P-values, confidence intervals, and evidence strength classification
- **Forensic Quality**: Complete chain of custody and court-admissible evidence packages
- **Performance**: Parallel processing with GraphCodeBERT, CodeT5, and MiniLM models

## AI Model Selection

| Model | Speed | Accuracy | Best Use Case | Memory |
|-------|-------|----------|---------------|--------|
| **graphcodebert** | Medium | **Highest** | **Forensic Analysis** | Medium |
| **codet5** | Slow | High | Cross-Language Detection | High |
| **mini** | **Fast** | Good | Large Repositories (1000+ files) | **Low** |

**Recommendation**: Use `graphcodebert` for forensic analysis requiring maximum accuracy and court-admissible evidence.

## Performance Metrics

- **Accuracy**: 90-100% strong evidence rate for cross-language detection
- **Obfuscation Resistance**: 94-96% similarity detection despite code modifications
- **Statistical Rigor**: p < 0.05 significance threshold, < 5% false positive rate
- **Speed**: < 60 seconds for comprehensive analysis, linear scaling with parallel processing
- **Languages**: 15+ programming languages with full AST support

## Installation & Usage

### Setup
```bash
# Install dependencies
pip install numpy scipy tqdm sentence-transformers transformers torch
pip install tree-sitter==0.20.4 tree-sitter-languages==1.9.1
```
**Requirements**: Python 3.8+, PyTorch 1.9+, Transformers 4.20+

### Basic Usage
```bash
# Standard forensic analysis
python SIPCompare.py --repoA /path/to/suspected --repoB /path/to/original \
                     --threshold 0.6 --embedding-model graphcodebert \
                     --parallel 4 --output evidence.zip

# Cross-language detection
python SIPCompare.py --repoA /path/to/python_repo --repoB /path/to/java_repo \
                     --cross-language --embedding-model codet5
```

### Key Options
- `--repoA/--repoB`: Repository paths (required)
- `--threshold`: Similarity threshold 0-1 (default: 0.75)
- `--embedding-model`: mini/graphcodebert/codet5 (default: graphcodebert)
- `--parallel`: Number of processes (default: 1)
- `--output`: Evidence package filename (default: evidence_package.zip)

## Supported Languages

**Full Support**: Python, Java, C/C++, JavaScript, TypeScript, Go, Rust, C#, PHP, Ruby, Swift, Kotlin, Scala  
**Semantic Only**: Shell Scripts, PowerShell

## Clone Detection & Evidence Classification

| Clone Type | Description | Evidence Level |
|------------|-------------|----------------|
| **Type 1** | Exact clones (whitespace/comments differ) | STRONG (>0.95) |
| **Type 2** | Renamed identifiers | STRONG (>0.85) |
| **Type 3** | Near-miss (added/deleted statements) | MODERATE (>0.75) |
| **Type 4** | Semantic clones (different syntax, same function) | MODERATE (>0.65) |

## Evidence Packages

**Contents**: Interactive HTML report, CSV/JSON data, executive summary, source code snapshots, chain of custody documentation

**Interpretation**:
- **STRONG**: High confidence IP theft (legal action recommended)
- **MODERATE**: Significant similarity (further investigation needed)  
- **WEAK**: Low similarity (likely coincidental)

## Troubleshooting

**Common Issues**:
- **"No processable files found"**: Check file extensions and repository paths
- **"Model loading failed"**: Install dependencies, check internet connection  
- **"Out of memory"**: Reduce parallel workers, use 'mini' model, process smaller batches
- **"Tree-sitter not available"**: Install with `pip install tree-sitter==0.20.4 tree-sitter-languages==1.9.1`

**Performance Tuning**:
- Large repositories: Use `--parallel 8 --embedding-model mini`
- Maximum accuracy: Use `--threshold 0.6 --embedding-model graphcodebert`
- Cross-language: Use `--embedding-model codet5 --cross-language`

## Forensic Considerations

- **Chain of Custody**: Complete audit trail with hash verification
- **Reproducible Results**: Deterministic analysis for court admissibility  
- **Privacy & Security**: Local processing, encrypted evidence packages

## License

This software is proprietary and confidential. Unauthorized use, distribution, or modification is strictly prohibited.

© 2025 Shane D. Shook, All Rights Reserved

**Disclaimer**: This tool is designed for legitimate intellectual property protection and forensic analysis. Users are responsible for ensuring compliance with applicable laws and regulations. The authors assume no liability for misuse of this software.
