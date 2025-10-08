# SIPCompare v2.0: Advanced Forensic Code Similarity Analysis Tool

## Overview

SIPCompare (Software Intellectual Property Compare) is a forensic tool designed to detect software intellectual property theft and code plagiarism using AI models, multi-dimensional analysis, and statistical validation to provide court-admissible evidence of code similarity across multiple programming languages.

## Core Features

### 🔍 **Advanced Detection Capabilities**
- **Multi-dimensional Analysis**: Combines token-based, semantic, structural, and control-flow similarity
- **Obfuscation Resistance**: Detects similarities despite variable renaming, code reordering, and style changes
- **Cross-Language Support**: Supports 15+ programming languages with semantic analysis
- **Clone Type Classification**: Industry-standard Type 1-4 clone detection

### 📊 **Statistical Rigor & Forensic Quality**
- **Statistical Significance Testing**: P-values and confidence intervals for all matches
- **Evidence Strength Classification**: STRONG/MODERATE/WEAK evidence categories
- **Chain of Custody**: Complete audit trail for legal proceedings
- **Comprehensive Evidence Packages**: ZIP archives with all analysis data

### 🚀 **Performance & AI Models**
- **Parallel Processing**: Multi-core support for large repositories
- **Advanced Embeddings**: GraphCodeBERT, CodeT5, and MiniLM support
- **AST-Based Analysis**: Tree-sitter parsers for accurate structural analysis
- **Memory Efficient**: Optimized for large codebases

## AI Models and Selection Guide

| Model | Speed | Accuracy | Best For | Cross-Language | Memory Usage |
|-------|-------|----------|----------|----------------|--------------|
| **graphcodebert** | Medium | **Highest** | **Forensic Analysis** | **Excellent** | Medium |
| **codet5** | Slow | High | Code Translation Detection | Very Good | High |
| **mini** | **Fast** | Good | Large Repositories | Good | **Low** |

**Model Selection:**
- **Forensic Analysis**: Use `graphcodebert` for maximum accuracy and court-admissible evidence
- **Large Repositories**: Use `mini` for faster processing of 1000+ files
- **Cross-Language Detection**: Use `graphcodebert` or `codet5` for Python↔Java, C++↔Python cases
- **Resource-Constrained**: Use `mini` for limited memory environments

## Analysis Workflow

1. **Code Collection**: Recursively scans repositories, detects languages, normalizes code
2. **Feature Extraction**: Extracts lexical, syntactic, semantic, and structural features
3. **Multi-Dimensional Analysis**: Combines token-based, AST-based, semantic, and control-flow similarity
4. **Statistical Validation**: Establishes baselines and calculates significance
5. **Evidence Classification**: Categorizes findings as STRONG/MODERATE/WEAK with obfuscation detection
6. **Forensic Reporting**: Generates comprehensive evidence packages

## Validation Results

### Key Performance Metrics
- **Cross-Language Detection**: 90-100% strong evidence rate for Python↔Java and C++↔Python
- **Obfuscation Resistance**: 94-96% semantic similarity maintained despite identifier renaming, code reordering, and structural changes
- **Statistical Rigor**: All matches p < 0.05 significance threshold, < 5% false positive rate
- **Processing Speed**: < 60 seconds for comprehensive test suite, linear scaling with parallel processing
- **Language Support**: 15+ programming languages with Tree-sitter AST parsing

### Validated Capabilities
- **Semantic Clone Detection**: Accurate Type 1-4 clone classification across multiple languages
- **Obfuscation Detection**: Identifies function renaming, structural modifications, and deliberate code changes
- **Forensic Quality**: Reproducible results with complete audit trail and court-admissible evidence packages

## Installation & Quick Start

### Prerequisites
```bash
# Install dependencies
pip install numpy scipy tqdm sentence-transformers transformers torch
pip install tree-sitter==0.20.4 tree-sitter-languages==1.9.1  # Enhanced AST analysis
```

**Requirements**: Python 3.8+, PyTorch 1.9+, Transformers 4.20+

### Usage Examples

```bash
# Basic analysis
python SIPCompare.py --repoA /path/to/repo1 --repoB /path/to/repo2

# Forensic analysis (recommended)
python SIPCompare.py --repoA /path/to/suspected --repoB /path/to/original \
                     --threshold 0.6 --embedding-model graphcodebert \
                     --parallel 4 --verbose --output evidence.zip

# Cross-language detection
python SIPCompare.py --repoA /path/to/python_repo --repoB /path/to/java_repo \
                     --cross-language --embedding-model codet5
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--repoA/--repoB` | Paths to repositories | Required |
| `--threshold` | Similarity threshold (0-1) | 0.75 |
| `--embedding-model` | mini/graphcodebert/codet5 | graphcodebert |
| `--parallel` | Number of parallel processes | 1 |
| `--cross-language` | Enable cross-language detection | False |
| `--output` | Output evidence package | evidence_package.zip |
| `--verbose` | Enable verbose logging | False |

## Supported Languages

**Full AST + Semantic Support**: Python, Java, C/C++, JavaScript, TypeScript, Go, Rust, C#, PHP, Ruby, Swift, Kotlin, Scala

**Semantic Only**: Shell Scripts (.sh), PowerShell (.ps1)

## Clone Type Classification

| Type | Description | Detection Method | Evidence Level |
|------|-------------|------------------|----------------|
| **Type 1** | Exact clones (whitespace/comments differ) | Hash-based comparison | STRONG (>0.95) |
| **Type 2** | Renamed identifiers | AST + identifier normalization | STRONG (>0.85) |
| **Type 3** | Near-miss (added/deleted statements) | LCS + structural analysis | MODERATE (>0.75) |
| **Type 4** | Semantic clones (different syntax, same function) | Semantic embeddings | MODERATE (>0.65) |

## Output & Evidence Packages

### Evidence Package Contents
- **Interactive HTML Report**: Forensic analysis with side-by-side comparisons
- **CSV/JSON Data**: Machine-readable analysis results
- **Executive Summary**: Non-technical overview for legal teams
- **Source Code Files**: Complete repository snapshots
- **Chain of Custody**: Forensic documentation trail

### Evidence Interpretation
- **STRONG**: High confidence IP theft (legal action recommended)
- **MODERATE**: Significant similarity (further investigation needed)  
- **WEAK**: Low similarity (likely coincidental)
- **Statistical Significance**: p < 0.05 threshold with confidence intervals
- **Obfuscation Patterns**: Identifier renaming, structural changes, control flow modifications

## Advanced Capabilities

### Technical Features
- **Code Normalization**: Identifier canonicalization, construct normalization, comment/string removal
- **Structural Analysis**: AST depth calculation, control flow extraction, complexity metrics
- **Semantic Analysis**: Transformer models with contextual understanding and cross-language capability
- **Performance Optimization**: Embedding caching, parallel processing, memory-efficient streaming

### Scalability & Performance
- **Large Repositories**: Tested on 10,000+ files, handles GB-sized codebases
- **Parallel Processing**: Multi-core utilization with optimized algorithms
- **Memory Management**: Efficient caching and garbage collection

## Legal & Forensic Considerations

### Forensic Integrity
- **Chain of Custody**: Complete audit trail with hash verification
- **Reproducible Results**: Deterministic analysis for court admissibility
- **Privacy & Security**: Local processing, encrypted evidence packages, restricted access

## Troubleshooting

### Common Issues & Solutions
- **"No processable files found"**: Check file extensions and repository paths
- **"Model loading failed"**: Install dependencies, check internet connection
- **"Out of memory"**: Reduce parallel workers, use 'mini' model, process smaller sets
- **"Tree-sitter not available"**: Install with `pip install tree-sitter==0.20.4 tree-sitter-languages==1.9.1`

### Performance Tuning Examples
```bash
# Large repositories (speed optimized)
python SIPCompare.py --repoA large_repo1 --repoB large_repo2 --parallel 8 --embedding-model mini

# High accuracy forensic analysis
python SIPCompare.py --repoA repo1 --repoB repo2 --threshold 0.6 --embedding-model graphcodebert

# Cross-language detection
python SIPCompare.py --repoA python_repo --repoB java_repo --embedding-model codet5 --cross-language
```

## Contributing

### Development Setup
```bash
git clone <repository-url>
cd SIPCompare
pip install -r requirements.txt
python -m pytest tests/
```

### Standards
- **Code Quality**: Type hints, comprehensive docstrings, PEP 8 compliance
- **Testing**: Unit and integration tests required
- **Feature Requests**: Open GitHub issues with use cases and performance considerations

## License

This software is proprietary and confidential. Unauthorized use, distribution, or modification is strictly prohibited.

© 2025 Shane D. Shook, All Rights Reserved

**Disclaimer**: This tool is designed for legitimate intellectual property protection and forensic analysis. Users are responsible for ensuring compliance with applicable laws and regulations. The authors assume no liability for misuse of this software.
