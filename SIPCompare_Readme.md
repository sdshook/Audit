# SIPCompare v2.0: Forensic Code Similarity Analysis Tool

## Overview

SIPCompare detects software intellectual property theft and code plagiarism using AI models, multi-dimensional analysis, and statistical validation to provide court-admissible evidence across 15+ programming languages.

**Key Capabilities**: Multi-dimensional similarity detection, obfuscation resistance, cross-language support, statistical validation (p < 0.05), forensic-quality evidence packages with complete chain of custody.

## AI Models & Performance

| Model | Speed | Accuracy | Best Use Case | Memory | Detection Rate |
|-------|-------|----------|---------------|--------|----------------|
| **graphcodebert** | Medium | **Highest** | **Forensic Analysis** | Medium | 90-100% |
| **codet5** | Slow | High | Cross-Language Detection | High | 90-95% |
| **mini** | **Fast** | Good | Large Repositories (1000+ files) | **Low** | 85-90% |

**Obfuscation Resistance**: 94-96% detection despite code modifications  
**Statistical Rigor**: p < 0.05 significance, < 5% false positive rate

## Installation & Usage

**Requirements**: Python 3.8+, PyTorch 1.9+, Transformers 4.20+

```bash
# Install dependencies
pip install numpy scipy tqdm sentence-transformers transformers torch
pip install tree-sitter==0.20.4 tree-sitter-languages==1.9.1
```

### Usage Examples
```bash
# Standard forensic analysis
python SIPCompare.py --repoA /path/to/suspected --repoB /path/to/original \
                     --threshold 0.6 --embedding-model graphcodebert \
                     --parallel 4 --output evidence.zip

# Cross-language detection
python SIPCompare.py --repoA /path/to/python_repo --repoB /path/to/java_repo \
                     --cross-language --embedding-model codet5
```

**Key Options**: `--repoA/--repoB` (required paths), `--threshold` (0-1, default: 0.75), `--embedding-model` (mini/graphcodebert/codet5), `--parallel` (processes, default: 1), `--output` (evidence filename)

## Supported Languages

**Full Support**: Python, Java, C/C++, JavaScript, TypeScript, Go, Rust, C#, PHP, Ruby, Swift, Kotlin, Scala  
**Semantic Only**: Shell Scripts, PowerShell

## Clone Detection & Evidence

| Clone Type | Description | Evidence Level |
|------------|-------------|----------------|
| **Type 1** | Exact clones (whitespace/comments differ) | STRONG (>0.95) |
| **Type 2** | Renamed identifiers | STRONG (>0.85) |
| **Type 3** | Near-miss (added/deleted statements) | MODERATE (>0.75) |
| **Type 4** | Semantic clones (different syntax, same function) | MODERATE (>0.65) |

**Evidence Package Contents**: Interactive HTML report, CSV/JSON data, executive summary, source code snapshots, chain of custody documentation with hash verification and complete audit trail for court admissibility.

## Troubleshooting & Performance

**Common Issues**:
- **"No processable files found"**: Check file extensions and repository paths
- **"Model loading failed"**: Install dependencies, check internet connection  
- **"Out of memory"**: Reduce parallel workers, use 'mini' model, process smaller batches
- **"Tree-sitter not available"**: Install with `pip install tree-sitter==0.20.4 tree-sitter-languages==1.9.1`

**Optimization**:
- **Large repositories**: `--parallel 8 --embedding-model mini`
- **High accuracy**: `--threshold 0.6 --embedding-model graphcodebert`
- **Cross-language**: `--embedding-model codet5 --cross-language`

## License

This software is proprietary and confidential. Unauthorized use, distribution, or modification is strictly prohibited.

© 2025 Shane D. Shook, All Rights Reserved

**Disclaimer**: This tool is designed for legitimate intellectual property protection and forensic analysis. Users are responsible for ensuring compliance with applicable laws and regulations. The authors assume no liability for misuse of this software.
