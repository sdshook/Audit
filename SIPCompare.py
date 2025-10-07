#!/usr/bin/env python3
"""
SIPCompare: Advanced Forensic Multi-Language Semantic Code Similarity Tool
© 2025 Shane D. Shook, All Rights Reserved

A comprehensive tool for detecting software intellectual property theft and code plagiarism
with advanced obfuscation resistance, statistical analysis, and forensic-quality reporting.

Example CLI usage:
    python SIPCompare.py --repoA /path/to/repo1 --repoB /path/to/repo2 --threshold 0.8 --embedding-model graphcodebert --output evidence_package.zip --parallel 4

Options:
    --repoA            Path to first repository
    --repoB            Path to second repository  
    --threshold        Similarity threshold (0-1), default 0.75
    --embedding-model  'mini', 'graphcodebert', or 'codet5', default 'graphcodebert'
    --output           Output ZIP forensic evidence package, default 'evidence_package.zip'
    --parallel         Number of parallel processes, default 1
    --granularity      Analysis granularity: 'file', 'function', 'block', default 'file'
    --statistical      Enable statistical significance testing, default True
    --cross-language   Enable cross-language detection, default False
"""

import os, re, hashlib, json, csv, datetime, zipfile, logging, warnings
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm

# Import torch for embedding models
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
import difflib
import pickle
import multiprocessing as mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional scipy import
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("SciPy not available, statistical analysis will be limited")

# Suppress warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------
# Data Structures
# ----------------------------
@dataclass
class CodeFeatures:
    """Comprehensive code features for similarity analysis"""
    path: str
    language: str
    raw_code: str
    normalized_code: str
    sha256: str
    md5: str
    
    # Structural features
    ast_structure: Dict[str, Any]
    control_flow_patterns: List[str]
    data_flow_patterns: List[str]
    call_sequences: List[str]
    
    # Metadata
    imports: int
    functions: int
    classes: int
    variables: int
    complexity: int
    lines_of_code: int
    
    # Embeddings
    semantic_embedding: Optional[np.ndarray]
    structural_embedding: Optional[np.ndarray]
    
    # Normalized identifiers
    identifier_map: Dict[str, str]
    canonical_identifiers: Set[str]

@dataclass
class SimilarityResult:
    """Comprehensive similarity analysis result"""
    file_a: str
    file_b: str
    clone_type: int  # 1-4 classification
    overall_similarity: float
    
    # Component similarities
    structural_similarity: float
    semantic_similarity: float
    token_similarity: float
    control_flow_similarity: float
    data_flow_similarity: float
    functional_similarity: float
    
    # Statistical measures
    confidence_interval: Tuple[float, float]
    p_value: float
    statistical_significance: bool
    
    # Forensic data
    evidence_strength: str  # "STRONG", "MODERATE", "WEAK"
    obfuscation_detected: bool
    transformation_patterns: List[str]

# ----------------------------
# Enhanced Semantic Embeddings
# ----------------------------
class EmbeddingManager:
    """Manages multiple embedding models for code analysis"""
    
    def __init__(self):
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load available embedding models"""
        try:
            from sentence_transformers import SentenceTransformer
            self.models['mini'] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Loaded MiniLM model")
        except ImportError:
            logger.warning("SentenceTransformers not available")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            self.models['graphcodebert_tokenizer'] = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
            self.models['graphcodebert'] = AutoModel.from_pretrained("microsoft/graphcodebert-base")
            logger.info("Loaded GraphCodeBERT model")
        except Exception as e:
            logger.warning(f"GraphCodeBERT not available: {e}")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            self.models['codet5_tokenizer'] = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
            self.models['codet5'] = AutoModel.from_pretrained("Salesforce/codet5-base")
            logger.info("Loaded CodeT5 model")
        except Exception as e:
            logger.warning(f"CodeT5 not available: {e}")
    
    def get_embedding(self, code: str, model_type: str = 'graphcodebert') -> np.ndarray:
        """Get semantic embedding for code"""
        if not TORCH_AVAILABLE:
            logger.warning("Torch not available, using fallback embeddings")
            return np.random.normal(0, 0.1, 768)  # Fallback embedding
            
        if model_type == 'mini' and 'mini' in self.models:
            return self.models['mini'].encode(code)
        
        elif model_type == 'graphcodebert' and 'graphcodebert' in self.models:
            tokenizer = self.models['graphcodebert_tokenizer']
            model = self.models['graphcodebert']
            
            # Handle long sequences by chunking
            max_length = 512
            tokens = tokenizer.tokenize(code)
            
            if len(tokens) <= max_length - 2:  # Account for special tokens
                inputs = tokenizer(code, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            else:
                # Chunk and average embeddings
                chunks = [tokens[i:i+max_length-2] for i in range(0, len(tokens), max_length-2)]
                embeddings = []
                
                for chunk in chunks:
                    chunk_text = tokenizer.convert_tokens_to_string(chunk)
                    inputs = tokenizer(chunk_text, return_tensors='pt', truncation=True, max_length=max_length, padding=True)
                    with torch.no_grad():
                        outputs = model(**inputs)
                        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
                
                return np.mean(embeddings, axis=0)
        
        elif model_type == 'codet5' and 'codet5' in self.models:
            tokenizer = self.models['codet5_tokenizer']
            model = self.models['codet5']
            inputs = tokenizer(code, return_tensors='pt', truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        else:
            logger.warning(f"Model {model_type} not available, returning zero vector")
            return np.zeros(768)

# Global embedding manager
embedding_manager = EmbeddingManager()

# ----------------------------
# Enhanced Tree-Sitter Setup and AST Analysis
# ----------------------------
try:
    from tree_sitter import Language, Parser
    import tree_sitter_languages
    TREE_SITTER_AVAILABLE = True
    logger.info("Tree-sitter available with tree-sitter-languages package")
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("Tree-sitter not available, falling back to regex-based analysis")

# Extended language support - mapping file extensions to tree-sitter-languages names
LANGUAGES = {
    '.py': 'python',
    '.cpp': 'cpp',
    '.c': 'c',
    '.h': 'c',
    '.hpp': 'cpp',
    '.java': 'java',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.go': 'go',
    '.rs': 'rust',
    '.cs': 'c_sharp',
    '.php': 'php',
    '.rb': 'ruby',
    '.swift': 'swift',
    '.kt': 'kotlin',
    '.scala': 'scala',
    # Scripts and markup
    '.sh': None,
    '.zsh': None,
    '.ksh': None,
    '.ps1': None,
    '.bat': None,
    '.cmd': None,
    '.md': None,
    '.txt': None,
    '.sql': None,
    '.xml': None,
    '.html': None,
    '.css': None,
}

class ASTAnalyzer:
    """Advanced AST analysis with structural feature extraction"""
    
    def __init__(self):
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._setup_parsers()
    
    def _setup_parsers(self):
        """Setup Tree-sitter parsers for supported languages using tree-sitter-languages"""
        try:
            # Initialize parsers using tree-sitter-languages
            for ext, lang_name in LANGUAGES.items():
                if lang_name:
                    try:
                        language = tree_sitter_languages.get_language(lang_name)
                        parser = Parser()
                        parser.set_language(language)
                        self.parsers[ext] = parser
                        logger.debug(f"Successfully loaded parser for {ext} ({lang_name})")
                    except Exception as e:
                        logger.warning(f"Failed to load parser for {ext} ({lang_name}): {e}")
                        self.parsers[ext] = None
                else:
                    self.parsers[ext] = None
            
            logger.info(f"Successfully initialized {len([p for p in self.parsers.values() if p])} Tree-sitter parsers")
        except Exception as e:
            logger.error(f"Failed to setup parsers: {e}")
            self.parsers = {}
    
    def extract_structural_features(self, code: str, language: str) -> Dict[str, Any]:
        """Extract comprehensive structural features from code"""
        parser = self.parsers.get(language)
        
        if parser and TREE_SITTER_AVAILABLE:
            return self._extract_ast_features(code, parser)
        else:
            return self._extract_regex_features(code, language)
    
    def _extract_ast_features(self, code: str, parser) -> Dict[str, Any]:
        """Extract features using Tree-sitter AST"""
        try:
            tree = parser.parse(bytes(code, 'utf8'))
            root = tree.root_node
            
            features = {
                'ast_depth': self._calculate_ast_depth(root),
                'node_types': self._extract_node_types(root),
                'control_structures': self._extract_control_structures(root, code),
                'function_signatures': self._extract_function_signatures(root, code),
                'variable_declarations': self._extract_variable_declarations(root, code),
                'call_patterns': self._extract_call_patterns(root, code),
                'structural_hash': self._compute_structural_hash(root),
                'complexity_metrics': self._calculate_complexity(root, code)
            }
            
            return features
        except Exception as e:
            logger.warning(f"AST extraction failed: {e}")
            return self._extract_regex_features(code, parser)
    
    def _extract_regex_features(self, code: str, language: str) -> Dict[str, Any]:
        """Fallback regex-based feature extraction"""
        features = {
            'control_structures': self._regex_control_structures(code),
            'function_patterns': self._regex_function_patterns(code, language),
            'variable_patterns': self._regex_variable_patterns(code, language),
            'import_patterns': self._regex_import_patterns(code, language),
            'complexity_estimate': self._estimate_complexity(code)
        }
        
        return features
    
    def _calculate_ast_depth(self, node, current_depth=0) -> int:
        """Calculate maximum depth of AST"""
        if not node.children:
            return current_depth
        
        max_child_depth = 0
        for child in node.children:
            child_depth = self._calculate_ast_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _extract_node_types(self, node) -> Counter:
        """Extract and count AST node types"""
        node_types = Counter()
        
        def traverse(n):
            node_types[n.type] += 1
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return node_types
    
    def _extract_control_structures(self, node, code: str) -> List[str]:
        """Extract control flow structures"""
        control_structures = []
        
        def traverse(n):
            if n.type in ['if_statement', 'while_statement', 'for_statement', 
                         'switch_statement', 'try_statement', 'with_statement']:
                structure_code = code[n.start_byte:n.end_byte]
                # Normalize the structure
                normalized = re.sub(r'\s+', ' ', structure_code[:100])  # First 100 chars
                control_structures.append(f"{n.type}:{normalized}")
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return control_structures
    
    def _extract_function_signatures(self, node, code: str) -> List[str]:
        """Extract function signatures"""
        signatures = []
        
        def traverse(n):
            if n.type in ['function_definition', 'method_definition', 'function_declaration']:
                sig_code = code[n.start_byte:n.end_byte]
                # Extract just the signature part
                lines = sig_code.split('\n')
                signature = lines[0] if lines else sig_code[:100]
                signatures.append(re.sub(r'\s+', ' ', signature.strip()))
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return signatures
    
    def _extract_variable_declarations(self, node, code: str) -> List[str]:
        """Extract variable declarations"""
        declarations = []
        
        def traverse(n):
            if n.type in ['variable_declaration', 'assignment', 'augmented_assignment']:
                decl_code = code[n.start_byte:n.end_byte]
                # Normalize declaration
                normalized = re.sub(r'\s+', ' ', decl_code.split('\n')[0][:50])
                declarations.append(normalized)
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return declarations
    
    def _extract_call_patterns(self, node, code: str) -> List[str]:
        """Extract function call patterns"""
        calls = []
        
        def traverse(n):
            if n.type in ['call', 'function_call', 'method_call']:
                call_code = code[n.start_byte:n.end_byte]
                # Extract function name and normalize
                call_normalized = re.sub(r'\s+', ' ', call_code.split('(')[0])
                calls.append(call_normalized)
            
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return calls
    
    def _compute_structural_hash(self, node) -> str:
        """Compute hash of AST structure (ignoring identifiers)"""
        def get_structure(n):
            if not n.children:
                return n.type
            return f"{n.type}({','.join(get_structure(child) for child in n.children)})"
        
        structure_str = get_structure(node)
        return hashlib.md5(structure_str.encode()).hexdigest()
    
    def _calculate_complexity(self, node, code: str) -> Dict[str, int]:
        """Calculate various complexity metrics"""
        metrics = {
            'cyclomatic_complexity': 1,  # Base complexity
            'nesting_depth': 0,
            'function_count': 0,
            'branch_count': 0
        }
        
        def traverse(n, depth=0):
            metrics['nesting_depth'] = max(metrics['nesting_depth'], depth)
            
            if n.type in ['if_statement', 'while_statement', 'for_statement', 
                         'case', 'catch_clause']:
                metrics['cyclomatic_complexity'] += 1
                metrics['branch_count'] += 1
            
            if n.type in ['function_definition', 'method_definition']:
                metrics['function_count'] += 1
            
            for child in n.children:
                traverse(child, depth + 1)
        
        traverse(node)
        return metrics
    
    def _regex_control_structures(self, code: str) -> List[str]:
        """Regex-based control structure extraction"""
        patterns = [
            r'\bif\s*\([^)]*\)',
            r'\bwhile\s*\([^)]*\)',
            r'\bfor\s*\([^)]*\)',
            r'\bswitch\s*\([^)]*\)',
            r'\btry\s*\{',
            r'\bcatch\s*\([^)]*\)'
        ]
        
        structures = []
        for pattern in patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            structures.extend(matches)
        
        return structures
    
    def _regex_function_patterns(self, code: str, language: str) -> List[str]:
        """Regex-based function pattern extraction"""
        if language == '.py':
            pattern = r'def\s+(\w+)\s*\([^)]*\):'
        elif language in ['.java', '.c', '.cpp', '.cs']:
            pattern = r'(?:public|private|protected|static)?\s*\w+\s+(\w+)\s*\([^)]*\)\s*\{'
        elif language == '.js':
            pattern = r'function\s+(\w+)\s*\([^)]*\)|(\w+)\s*=\s*function'
        else:
            pattern = r'function\s+(\w+)|def\s+(\w+)|(\w+)\s*\([^)]*\)\s*\{'
        
        return re.findall(pattern, code, re.IGNORECASE)
    
    def _regex_variable_patterns(self, code: str, language: str) -> List[str]:
        """Regex-based variable pattern extraction"""
        if language == '.py':
            pattern = r'(\w+)\s*=\s*[^=]'
        elif language in ['.java', '.c', '.cpp', '.cs']:
            pattern = r'(?:int|string|double|float|bool|var)\s+(\w+)'
        elif language == '.js':
            pattern = r'(?:var|let|const)\s+(\w+)'
        else:
            pattern = r'(\w+)\s*=\s*[^=]'
        
        return re.findall(pattern, code, re.IGNORECASE)
    
    def _regex_import_patterns(self, code: str, language: str) -> List[str]:
        """Regex-based import pattern extraction"""
        if language == '.py':
            pattern = r'(?:import|from)\s+([^\s;]+)'
        elif language == '.java':
            pattern = r'import\s+([^;]+);'
        elif language == '.js':
            pattern = r'(?:import|require)\s*\(?[\'"]([^\'"]+)[\'"]'
        else:
            pattern = r'(?:import|include|using)\s+([^\s;]+)'
        
        return re.findall(pattern, code, re.IGNORECASE)
    
    def _estimate_complexity(self, code: str) -> int:
        """Estimate cyclomatic complexity using regex"""
        complexity_keywords = [
            r'\bif\b', r'\bwhile\b', r'\bfor\b', r'\bswitch\b',
            r'\bcatch\b', r'\bcase\b', r'\b\?\s*:', r'\b&&\b', r'\b\|\|\b'
        ]
        
        complexity = 1  # Base complexity
        for pattern in complexity_keywords:
            complexity += len(re.findall(pattern, code, re.IGNORECASE))
        
        return complexity

# Global AST analyzer
ast_analyzer = ASTAnalyzer()

# ----------------------------
# Advanced Code Normalization and Obfuscation Resistance
# ----------------------------
class CodeNormalizer:
    """Advanced code normalization to resist obfuscation techniques"""
    
    def __init__(self):
        self.identifier_counter = 0
        self.reserved_words = {
            '.py': {'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except', 'import', 'from', 'return'},
            '.java': {'public', 'private', 'class', 'interface', 'if', 'else', 'for', 'while', 'try', 'catch', 'import', 'return'},
            '.js': {'function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'try', 'catch', 'import', 'return'},
            '.c': {'int', 'char', 'float', 'double', 'if', 'else', 'for', 'while', 'return', 'include'},
            '.cpp': {'int', 'char', 'float', 'double', 'class', 'if', 'else', 'for', 'while', 'return', 'include'},
        }
    
    def normalize_code(self, code: str, language: str) -> Tuple[str, Dict[str, str]]:
        """Comprehensive code normalization with identifier mapping"""
        # Step 1: Remove comments and strings
        code_no_comments = self._remove_comments_and_strings(code, language)
        
        # Step 2: Normalize whitespace
        code_normalized = re.sub(r'\s+', ' ', code_no_comments).strip()
        
        # Step 3: Extract and normalize identifiers
        identifier_map = self._extract_identifiers(code_normalized, language)
        code_with_canonical_ids = self._replace_identifiers(code_normalized, identifier_map, language)
        
        # Step 4: Normalize equivalent constructs
        code_canonical = self._normalize_constructs(code_with_canonical_ids, language)
        
        # Step 5: Sort and normalize imports/includes
        code_final = self._normalize_imports(code_canonical, language)
        
        return code_final, identifier_map
    
    def _remove_comments_and_strings(self, code: str, language: str) -> str:
        """Remove comments and string literals while preserving structure"""
        if language == '.py':
            # Remove Python comments and strings
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            code = re.sub(r'"""[\s\S]*?"""', '""', code)
            code = re.sub(r"'''[\s\S]*?'''", "''", code)
            code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
            code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
        
        elif language in ['.java', '.c', '.cpp', '.cs', '.js']:
            # Remove C-style comments and strings
            code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
            code = re.sub(r'/\*[\s\S]*?\*/', '', code)
            code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
            code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
        
        elif language in ['.sh', '.zsh', '.ksh']:
            # Remove shell comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            code = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', '""', code)
            code = re.sub(r"'[^'\\]*(?:\\.[^'\\]*)*'", "''", code)
        
        elif language == '.ps1':
            # Remove PowerShell comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            code = re.sub(r'<#[\s\S]*?#>', '', code)
            code = re.sub(r'"[^"]*"', '""', code)
            code = re.sub(r"'[^']*'", "''", code)
        
        return code
    
    def _extract_identifiers(self, code: str, language: str) -> Dict[str, str]:
        """Extract and create canonical mapping for identifiers"""
        identifier_map = {}
        reserved = self.reserved_words.get(language, set())
        
        # Extract identifiers based on language
        if language == '.py':
            # Python identifiers: variables, functions, classes
            patterns = [
                r'\bdef\s+(\w+)',  # function definitions
                r'\bclass\s+(\w+)',  # class definitions
                r'\b(\w+)\s*=\s*[^=]',  # variable assignments
                r'\b(\w+)\s*\(',  # function calls
            ]
        elif language in ['.java', '.c', '.cpp', '.cs']:
            patterns = [
                r'\b(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(',  # methods
                r'\b(?:class|interface)\s+(\w+)',  # classes/interfaces
                r'\b(?:int|string|double|float|bool|var)\s+(\w+)',  # variables
                r'\b(\w+)\s*\(',  # function calls
            ]
        elif language == '.js':
            patterns = [
                r'\bfunction\s+(\w+)',  # function definitions
                r'\b(?:var|let|const)\s+(\w+)',  # variable declarations
                r'\b(\w+)\s*=\s*function',  # function expressions
                r'\b(\w+)\s*\(',  # function calls
            ]
        else:
            # Generic patterns
            patterns = [
                r'\b(\w+)\s*\(',  # function calls
                r'\b(\w+)\s*=',  # assignments
            ]
        
        # Extract all identifiers
        all_identifiers = set()
        for pattern in patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    all_identifiers.update(m for m in match if m and m not in reserved)
                elif match and match not in reserved:
                    all_identifiers.add(match)
        
        # Create canonical mapping
        sorted_identifiers = sorted(all_identifiers)
        for i, identifier in enumerate(sorted_identifiers):
            identifier_map[identifier] = f"ID_{i:03d}"
        
        return identifier_map
    
    def _replace_identifiers(self, code: str, identifier_map: Dict[str, str], language: str) -> str:
        """Replace identifiers with canonical names"""
        # Sort by length (longest first) to avoid partial replacements
        sorted_identifiers = sorted(identifier_map.keys(), key=len, reverse=True)
        
        for original, canonical in [(k, identifier_map[k]) for k in sorted_identifiers]:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(original) + r'\b'
            code = re.sub(pattern, canonical, code)
        
        return code
    
    def _normalize_constructs(self, code: str, language: str) -> str:
        """Normalize equivalent language constructs"""
        if language == '.py':
            # Normalize Python constructs
            code = re.sub(r'\bTrue\b', 'true', code)
            code = re.sub(r'\bFalse\b', 'false', code)
            code = re.sub(r'\bNone\b', 'null', code)
        
        elif language in ['.java', '.c', '.cpp', '.cs']:
            # Normalize C-style constructs
            code = re.sub(r'\btrue\b', 'true', code, flags=re.IGNORECASE)
            code = re.sub(r'\bfalse\b', 'false', code, flags=re.IGNORECASE)
            code = re.sub(r'\bnull\b', 'null', code, flags=re.IGNORECASE)
        
        elif language == '.js':
            # Normalize JavaScript constructs
            code = re.sub(r'\bundefined\b', 'null', code)
            code = re.sub(r'\btrue\b', 'true', code)
            code = re.sub(r'\bfalse\b', 'false', code)
        
        # Normalize common patterns across languages
        code = re.sub(r'!=', '!==', code)  # Normalize inequality
        code = re.sub(r'==', '===', code)  # Normalize equality
        
        return code
    
    def _normalize_imports(self, code: str, language: str) -> str:
        """Normalize and sort import statements"""
        lines = code.split('\n')
        imports = []
        other_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if language == '.py' and (line.startswith('import ') or line.startswith('from ')):
                imports.append(line)
            elif language in ['.java', '.c', '.cpp'] and line.startswith('import '):
                imports.append(line)
            elif language == '.js' and (line.startswith('import ') or 'require(' in line):
                imports.append(line)
            else:
                other_lines.append(line)
        
        # Sort imports and combine with other code
        imports.sort()
        normalized_lines = imports + [''] + other_lines if imports else other_lines
        
        return '\n'.join(normalized_lines)

# ----------------------------
# Enhanced Utility Functions
# ----------------------------
def file_hash(code: str) -> Tuple[str, str]:
    """Generate SHA256 and MD5 hashes for code"""
    code_bytes = code.encode('utf-8')
    sha256 = hashlib.sha256(code_bytes).hexdigest()
    md5 = hashlib.md5(code_bytes).hexdigest()
    return sha256, md5

def extract_comprehensive_metadata(code: str, language: str, structural_features: Dict[str, Any]) -> Dict[str, Any]:
    """Extract comprehensive metadata from code"""
    metadata = {
        'lines_of_code': len([line for line in code.split('\n') if line.strip()]),
        'total_lines': len(code.split('\n')),
        'imports': 0,
        'functions': 0,
        'classes': 0,
        'variables': 0,
        'complexity': 0,
        'entry_points': 0,
        'comments_ratio': 0.0,
    }
    
    # Language-specific metadata extraction
    if language == '.py':
        metadata['imports'] = len(re.findall(r'^\s*(?:import|from)\s+', code, re.MULTILINE))
        metadata['functions'] = len(re.findall(r'^\s*def\s+\w+', code, re.MULTILINE))
        metadata['classes'] = len(re.findall(r'^\s*class\s+\w+', code, re.MULTILINE))
        metadata['variables'] = len(re.findall(r'^\s*\w+\s*=\s*[^=]', code, re.MULTILINE))
        metadata['entry_points'] = len(re.findall(r'if\s+__name__\s*==\s*[\'"]__main__[\'"]', code))
    
    elif language in ['.java', '.c', '.cpp', '.cs']:
        metadata['imports'] = len(re.findall(r'^\s*(?:import|#include)\s+', code, re.MULTILINE))
        metadata['functions'] = len(re.findall(r'\b(?:public|private|protected)?\s*(?:static)?\s*\w+\s+\w+\s*\(', code))
        metadata['classes'] = len(re.findall(r'\b(?:class|interface)\s+\w+', code))
        metadata['variables'] = len(re.findall(r'\b(?:int|string|double|float|bool|var)\s+\w+', code))
        metadata['entry_points'] = len(re.findall(r'\bpublic\s+static\s+void\s+main\s*\(', code))
    
    elif language == '.js':
        metadata['imports'] = len(re.findall(r'^\s*(?:import|const\s+\w+\s*=\s*require)', code, re.MULTILINE))
        metadata['functions'] = len(re.findall(r'\b(?:function\s+\w+|\w+\s*=\s*function)', code))
        metadata['classes'] = len(re.findall(r'\bclass\s+\w+', code))
        metadata['variables'] = len(re.findall(r'\b(?:var|let|const)\s+\w+', code))
        metadata['entry_points'] = 1 if 'module.exports' in code or 'export' in code else 0
    
    # Extract from structural features if available
    if structural_features:
        if 'complexity_metrics' in structural_features:
            complexity_metrics = structural_features['complexity_metrics']
            if isinstance(complexity_metrics, dict):
                metadata['complexity'] = complexity_metrics.get('cyclomatic_complexity', 0)
            else:
                metadata['complexity'] = complexity_metrics
        
        if 'function_signatures' in structural_features:
            metadata['functions'] = max(metadata['functions'], len(structural_features['function_signatures']))
    
    # Calculate comments ratio
    total_chars = len(code)
    if total_chars > 0:
        comment_chars = len(re.findall(r'#.*|//.*|/\*[\s\S]*?\*/', code))
        metadata['comments_ratio'] = comment_chars / total_chars
    
    return metadata

def enhanced_jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
    """Enhanced Jaccard similarity with token weighting"""
    if not tokens_a or not tokens_b:
        return 0.0
    
    set_a, set_b = set(tokens_a), set(tokens_b)
    intersection = set_a & set_b
    union = set_a | set_b
    
    if not union:
        return 0.0
    
    # Weight tokens by frequency (rare tokens get higher weight)
    all_tokens = tokens_a + tokens_b
    token_freq = Counter(all_tokens)
    total_tokens = len(all_tokens)
    
    weighted_intersection = sum(1.0 / (token_freq[token] / total_tokens) for token in intersection)
    weighted_union = sum(1.0 / (token_freq[token] / total_tokens) for token in union)
    
    return weighted_intersection / weighted_union if weighted_union > 0 else 0.0

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    if vec_a is None or vec_b is None:
        return 0.0
    
    # Handle zero vectors
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(vec_a, vec_b) / (norm_a * norm_b)

def structural_similarity(features_a: Dict[str, Any], features_b: Dict[str, Any]) -> float:
    """Calculate structural similarity between code features"""
    if not features_a or not features_b:
        return 0.0
    
    similarities = []
    
    # Compare control structures
    if 'control_structures' in features_a and 'control_structures' in features_b:
        ctrl_a = set(features_a['control_structures'])
        ctrl_b = set(features_b['control_structures'])
        if ctrl_a or ctrl_b:
            ctrl_sim = len(ctrl_a & ctrl_b) / len(ctrl_a | ctrl_b)
            similarities.append(ctrl_sim)
    
    # Compare function signatures
    if 'function_signatures' in features_a and 'function_signatures' in features_b:
        func_a = set(features_a['function_signatures'])
        func_b = set(features_b['function_signatures'])
        if func_a or func_b:
            func_sim = len(func_a & func_b) / len(func_a | func_b)
            similarities.append(func_sim)
    
    # Compare AST structure hashes
    if 'structural_hash' in features_a and 'structural_hash' in features_b:
        hash_sim = 1.0 if features_a['structural_hash'] == features_b['structural_hash'] else 0.0
        similarities.append(hash_sim)
    
    # Compare complexity metrics
    if 'complexity_metrics' in features_a and 'complexity_metrics' in features_b:
        comp_a = features_a['complexity_metrics']
        comp_b = features_b['complexity_metrics']
        if isinstance(comp_a, dict) and isinstance(comp_b, dict):
            comp_similarities = []
            for key in set(comp_a.keys()) & set(comp_b.keys()):
                val_a, val_b = comp_a[key], comp_b[key]
                if val_a == 0 and val_b == 0:
                    comp_similarities.append(1.0)
                elif max(val_a, val_b) > 0:
                    comp_similarities.append(1.0 - abs(val_a - val_b) / max(val_a, val_b))
            
            if comp_similarities:
                similarities.append(np.mean(comp_similarities))
    
    return np.mean(similarities) if similarities else 0.0

# Global code normalizer
code_normalizer = CodeNormalizer()

# ----------------------------
# Clone Type Classification
# ----------------------------
class CloneDetector:
    """Industry-standard clone type classification (Type 1-4)"""
    
    def classify_clone_type(self, features_a: CodeFeatures, features_b: CodeFeatures) -> Tuple[int, List[str]]:
        """Classify clone type and detect transformation patterns"""
        transformations = []
        
        # Type 1: Exact clones (only whitespace/comment differences)
        if features_a.sha256 == features_b.sha256:
            return 1, ["exact_match"]
        
        # Check for exact match after normalization
        if features_a.normalized_code == features_b.normalized_code:
            transformations.append("whitespace_normalization")
            return 1, transformations
        
        # Type 2: Renamed clones (identifier changes)
        if self._is_type2_clone(features_a, features_b):
            transformations.append("identifier_renaming")
            if features_a.ast_structure.get('structural_hash') == features_b.ast_structure.get('structural_hash'):
                return 2, transformations
        
        # Type 3: Near-miss clones (statement additions/deletions)
        if self._is_type3_clone(features_a, features_b):
            transformations.extend(self._detect_type3_transformations(features_a, features_b))
            return 3, transformations
        
        # Type 4: Semantic clones (different syntax, same functionality)
        if self._is_type4_clone(features_a, features_b):
            transformations.extend(self._detect_type4_transformations(features_a, features_b))
            return 4, transformations
        
        return 0, []  # No clone detected
    
    def _is_type2_clone(self, features_a: CodeFeatures, features_b: CodeFeatures) -> bool:
        """Check if files are Type 2 clones (renamed identifiers)"""
        # Compare structural patterns ignoring identifiers
        struct_a = features_a.ast_structure
        struct_b = features_b.ast_structure
        
        if not struct_a or not struct_b:
            return False
        
        # Compare control flow patterns
        ctrl_a = set(struct_a.get('control_structures', []))
        ctrl_b = set(struct_b.get('control_structures', []))
        
        if ctrl_a and ctrl_b:
            ctrl_similarity = len(ctrl_a & ctrl_b) / len(ctrl_a | ctrl_b)
            return ctrl_similarity > 0.8
        
        return False
    
    def _is_type3_clone(self, features_a: CodeFeatures, features_b: CodeFeatures) -> bool:
        """Check if files are Type 3 clones (near-miss with modifications)"""
        # Use sequence alignment on normalized tokens
        tokens_a = features_a.normalized_code.split()
        tokens_b = features_b.normalized_code.split()
        
        if not tokens_a or not tokens_b:
            return False
        
        # Calculate longest common subsequence ratio
        lcs_ratio = self._lcs_ratio(tokens_a, tokens_b)
        return 0.6 <= lcs_ratio <= 0.9
    
    def _is_type4_clone(self, features_a: CodeFeatures, features_b: CodeFeatures) -> bool:
        """Check if files are Type 4 clones (semantic similarity)"""
        # Use semantic embeddings and functional similarity
        if features_a.semantic_embedding is not None and features_b.semantic_embedding is not None:
            semantic_sim = cosine_similarity(features_a.semantic_embedding, features_b.semantic_embedding)
            
            # Check if functions have similar complexity and structure
            complexity_sim = self._complexity_similarity(features_a, features_b)
            
            return semantic_sim > 0.7 and complexity_sim > 0.6
        
        return False
    
    def _detect_type3_transformations(self, features_a: CodeFeatures, features_b: CodeFeatures) -> List[str]:
        """Detect specific Type 3 transformations"""
        transformations = []
        
        # Compare line counts
        lines_a = features_a.lines_of_code
        lines_b = features_b.lines_of_code
        
        if lines_a != lines_b:
            if lines_a > lines_b:
                transformations.append("statement_deletion")
            else:
                transformations.append("statement_addition")
        
        # Compare function counts
        funcs_a = features_a.functions
        funcs_b = features_b.functions
        
        if funcs_a != funcs_b:
            transformations.append("function_modification")
        
        return transformations
    
    def _detect_type4_transformations(self, features_a: CodeFeatures, features_b: CodeFeatures) -> List[str]:
        """Detect specific Type 4 transformations"""
        transformations = []
        
        # Different languages
        if features_a.language != features_b.language:
            transformations.append("language_translation")
        
        # Different algorithmic approaches
        if features_a.complexity != features_b.complexity:
            transformations.append("algorithmic_change")
        
        # Different control structures
        ctrl_a = set(features_a.control_flow_patterns)
        ctrl_b = set(features_b.control_flow_patterns)
        
        if ctrl_a != ctrl_b:
            transformations.append("control_flow_change")
        
        return transformations
    
    def _lcs_ratio(self, seq_a: List[str], seq_b: List[str]) -> float:
        """Calculate longest common subsequence ratio"""
        m, n = len(seq_a), len(seq_b)
        if m == 0 or n == 0:
            return 0.0
        
        # Dynamic programming LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i-1] == seq_b[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        return lcs_length / max(m, n)
    
    def _complexity_similarity(self, features_a: CodeFeatures, features_b: CodeFeatures) -> float:
        """Calculate complexity similarity between features"""
        if features_a.complexity == 0 and features_b.complexity == 0:
            return 1.0
        
        max_complexity = max(features_a.complexity, features_b.complexity)
        if max_complexity == 0:
            return 1.0
        
        return 1.0 - abs(features_a.complexity - features_b.complexity) / max_complexity

# ----------------------------
# Statistical Analysis Framework
# ----------------------------
class StatisticalAnalyzer:
    """Statistical significance testing for similarity results"""
    
    def __init__(self):
        self.baseline_similarities = []
        self.similarity_distribution = None
    
    def establish_baseline(self, random_samples: List[Tuple[CodeFeatures, CodeFeatures]]):
        """Establish baseline similarity distribution from random code pairs"""
        logger.info("Establishing statistical baseline...")
        
        similarities = []
        for features_a, features_b in random_samples:
            # Calculate various similarity measures
            token_sim = enhanced_jaccard_similarity(
                features_a.normalized_code.split(),
                features_b.normalized_code.split()
            )
            
            semantic_sim = cosine_similarity(
                features_a.semantic_embedding,
                features_b.semantic_embedding
            )
            
            structural_sim = structural_similarity(
                features_a.ast_structure,
                features_b.ast_structure
            )
            
            # Combined similarity
            combined_sim = (token_sim + semantic_sim + structural_sim) / 3
            similarities.append(combined_sim)
        
        self.baseline_similarities = similarities
        
        # Fit distribution
        if similarities and SCIPY_AVAILABLE:
            self.similarity_distribution = stats.norm.fit(similarities)
            logger.info(f"Baseline established: μ={self.similarity_distribution[0]:.4f}, σ={self.similarity_distribution[1]:.4f}")
        elif similarities:
            # Fallback without scipy
            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)
            self.similarity_distribution = (mean_sim, std_sim)
            logger.info(f"Baseline established (no SciPy): μ={mean_sim:.4f}, σ={std_sim:.4f}")
    
    def calculate_significance(self, similarity_score: float) -> Tuple[float, bool]:
        """Calculate p-value and statistical significance with robust testing"""
        if not self.baseline_similarities:
            logger.warning("No baseline established, cannot calculate significance")
            return 1.0, False
        
        # Convert to numpy array for statistical operations
        baseline_array = np.array(self.baseline_similarities)
        
        if SCIPY_AVAILABLE:
            # Multiple statistical tests for robustness
            
            # 1. Z-test against baseline
            if len(self.baseline_similarities) > 30:  # Large sample
                mean, std = np.mean(baseline_array), np.std(baseline_array)
                if std > 0:
                    z_score = (similarity_score - mean) / std
                    p_value_z = 1 - stats.norm.cdf(z_score)
                else:
                    p_value_z = 1.0
            else:
                p_value_z = 1.0
            
            # 2. Mann-Whitney U test (non-parametric)
            # Compare single score against baseline distribution
            test_sample = [similarity_score]
            try:
                u_statistic, p_value_mw = stats.mannwhitneyu(
                    test_sample, baseline_array, alternative='greater'
                )
            except ValueError:
                p_value_mw = 1.0
            
            # 3. Percentile-based significance
            percentile = stats.percentileofscore(baseline_array, similarity_score)
            p_value_percentile = (100 - percentile) / 100
            
            # Combine p-values using Fisher's method for robustness
            p_values = [p for p in [p_value_z, p_value_mw, p_value_percentile] if p > 0]
            if len(p_values) >= 2:
                try:
                    combined_statistic = -2 * np.sum(np.log(p_values))
                    combined_p_value = 1 - stats.chi2.cdf(combined_statistic, 2 * len(p_values))
                except:
                    combined_p_value = min(p_values)
            else:
                combined_p_value = p_values[0] if p_values else 1.0
            
            # Conservative significance threshold for forensic quality
            is_significant = combined_p_value < 0.01  # More stringent than typical 0.05
            
            return combined_p_value, is_significant
        
        else:
            # Fallback without scipy - use percentile-based approach
            baseline_array = np.array(self.baseline_similarities)
            percentile = np.sum(baseline_array < similarity_score) / len(baseline_array)
            p_value = 1 - percentile
            
            # Conservative threshold
            is_significant = p_value < 0.01
            
            return p_value, is_significant
    
    def calculate_confidence_interval(self, similarity_score: float, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for similarity score"""
        if not self.similarity_distribution or not SCIPY_AVAILABLE:
            return (similarity_score, similarity_score)
        
        mean, std = self.similarity_distribution
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        z_critical = stats.norm.ppf(1 - alpha/2)
        
        margin_of_error = z_critical * std
        
        lower_bound = max(0.0, similarity_score - margin_of_error)
        upper_bound = min(1.0, similarity_score + margin_of_error)
        
        return (lower_bound, upper_bound)

# ----------------------------
# Enhanced Repository Processing
# ----------------------------
def collect_files_parallel(repo_path: str, embedding_model: str = 'graphcodebert', 
                          max_workers: int = 1) -> List[CodeFeatures]:
    """Collect and process files with parallel processing"""
    logger.info(f"Collecting files from {repo_path}")
    
    # Find all relevant files
    file_paths = []
    for root, _, files in os.walk(repo_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in LANGUAGES:
                file_paths.append(os.path.join(root, f))
    
    logger.info(f"Found {len(file_paths)} files to process")
    
    if max_workers == 1:
        # Sequential processing
        return [process_single_file(path, embedding_model) for path in tqdm(file_paths, desc="Processing files")]
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_file, path, embedding_model): path 
                      for path in file_paths}
            
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.warning(f"Failed to process {futures[future]}: {e}")
            
            return results

def process_single_file(file_path: str, embedding_model: str) -> Optional[CodeFeatures]:
    """Process a single file and extract all features"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_code = f.read()
        
        if not raw_code.strip():
            return None
        
        # Determine language
        language = os.path.splitext(file_path)[1].lower()
        
        # Normalize code
        normalized_code, identifier_map = code_normalizer.normalize_code(raw_code, language)
        
        # Generate hashes
        sha256, md5 = file_hash(normalized_code)
        
        # Extract structural features
        structural_features = ast_analyzer.extract_structural_features(raw_code, language)
        
        # Extract metadata
        metadata = extract_comprehensive_metadata(raw_code, language, structural_features)
        
        # Generate embeddings
        semantic_embedding = embedding_manager.get_embedding(normalized_code, embedding_model)
        structural_embedding = embedding_manager.get_embedding(
            str(structural_features.get('structural_hash', '')), embedding_model
        )
        
        # Extract patterns
        control_flow_patterns = structural_features.get('control_structures', [])
        data_flow_patterns = structural_features.get('variable_declarations', [])
        call_sequences = structural_features.get('call_patterns', [])
        
        return CodeFeatures(
            path=file_path,
            language=language,
            raw_code=raw_code,
            normalized_code=normalized_code,
            sha256=sha256,
            md5=md5,
            ast_structure=structural_features,
            control_flow_patterns=control_flow_patterns,
            data_flow_patterns=data_flow_patterns,
            call_sequences=call_sequences,
            imports=metadata['imports'],
            functions=metadata['functions'],
            classes=metadata['classes'],
            variables=metadata['variables'],
            complexity=metadata['complexity'],
            lines_of_code=metadata['lines_of_code'],
            semantic_embedding=semantic_embedding,
            structural_embedding=structural_embedding,
            identifier_map=identifier_map,
            canonical_identifiers=set(identifier_map.values())
        )
        
    except Exception as e:
        logger.warning(f"Failed to process {file_path}: {e}")
        return None

# ----------------------------
# Advanced Comparison Engine
# ----------------------------
class SimilarityEngine:
    """Advanced similarity analysis with multiple algorithms and statistical validation"""
    
    def __init__(self, statistical_analysis: bool = True, cross_language: bool = False):
        self.clone_detector = CloneDetector()
        self.statistical_analyzer = StatisticalAnalyzer() if statistical_analysis else None
        self.cross_language = cross_language
        
        # Language-aware similarity weights for forensic accuracy
        self.similarity_weights = {
            'same_language': {
                'token': 0.35,
                'semantic': 0.30,
                'structural': 0.25,
                'control_flow': 0.10
            },
            'cross_language': {
                'token': 0.05,        # Minimal weight for different languages
                'semantic': 0.60,     # Primary indicator for cross-language
                'functional': 0.25,   # Algorithm/logic similarity
                'structural': 0.10    # Normalized structure patterns
            }
        }
    
    def compare_repositories(self, repo_a_files: List[CodeFeatures], repo_b_files: List[CodeFeatures],
                           threshold: float = 0.75, enable_statistical: bool = True) -> List[SimilarityResult]:
        """Compare two repositories with comprehensive similarity analysis"""
        logger.info(f"Comparing {len(repo_a_files)} files from Repo A with {len(repo_b_files)} files from Repo B")
        
        # Establish statistical baseline if enabled
        if enable_statistical and self.statistical_analyzer:
            self._establish_baseline(repo_a_files, repo_b_files)
        
        matches = []
        total_comparisons = len(repo_a_files) * len(repo_b_files)
        
        # Filter out None values
        valid_files_a = [f for f in repo_a_files if f is not None]
        valid_files_b = [f for f in repo_b_files if f is not None]
        
        total_comparisons = len(valid_files_a) * len(valid_files_b)
        logger.info(f"Comparing {len(valid_files_a)} valid files from Repo A with {len(valid_files_b)} valid files from Repo B")
        
        with tqdm(total=total_comparisons, desc="Comparing files") as pbar:
            for file_a in valid_files_a:
                for file_b in valid_files_b:
                    similarity_result = self._compare_files(file_a, file_b, threshold)
                    if similarity_result and similarity_result.overall_similarity >= threshold:
                        matches.append(similarity_result)
                    pbar.update(1)
        
        logger.info(f"Found {len(matches)} matches above threshold {threshold}")
        return matches
    
    def _compare_files(self, file_a: CodeFeatures, file_b: CodeFeatures, threshold: float) -> Optional[SimilarityResult]:
        """Compare two files comprehensively with language-aware scoring"""
        try:
            # Calculate component similarities
            token_sim = self._calculate_token_similarity(file_a, file_b)
            semantic_sim = self._calculate_semantic_similarity(file_a, file_b)
            structural_sim = self._calculate_structural_similarity(file_a, file_b)
            control_flow_sim = self._calculate_control_flow_similarity(file_a, file_b)
            data_flow_sim = self._calculate_data_flow_similarity(file_a, file_b)
            functional_sim = self._calculate_functional_similarity(file_a, file_b)
            
            # Determine if this is cross-language comparison
            is_cross_language = (file_a.language != file_b.language) or self.cross_language
            
            # Select appropriate weights based on language comparison type
            if is_cross_language:
                weights = self.similarity_weights['cross_language']
                # Calculate weighted overall similarity for cross-language
                overall_sim = (
                    weights['token'] * token_sim +
                    weights['semantic'] * semantic_sim +
                    weights['functional'] * functional_sim +
                    weights['structural'] * structural_sim
                )
            else:
                weights = self.similarity_weights['same_language']
                # Calculate weighted overall similarity for same language
                overall_sim = (
                    weights['token'] * token_sim +
                    weights['semantic'] * semantic_sim +
                    weights['structural'] * structural_sim +
                    weights['control_flow'] * control_flow_sim
                )
            
            # Only proceed if above threshold
            if overall_sim < threshold:
                return None
            
            # Classify clone type and detect transformations
            clone_type, transformations = self.clone_detector.classify_clone_type(file_a, file_b)
            
            # Statistical analysis
            p_value, is_significant = (1.0, False)
            confidence_interval = (overall_sim, overall_sim)
            
            if self.statistical_analyzer:
                p_value, is_significant = self.statistical_analyzer.calculate_significance(overall_sim)
                confidence_interval = self.statistical_analyzer.calculate_confidence_interval(overall_sim)
            
            # Store semantic similarity for evidence classification
            self._last_semantic_similarity = semantic_sim
            
            # Determine evidence strength
            evidence_strength = self._determine_evidence_strength(
                overall_sim, clone_type, is_significant, transformations
            )
            
            # Detect obfuscation attempts
            obfuscation_detected = self._detect_obfuscation(file_a, file_b, transformations)
            
            return SimilarityResult(
                file_a=file_a.path,
                file_b=file_b.path,
                clone_type=clone_type,
                overall_similarity=overall_sim,
                structural_similarity=structural_sim,
                semantic_similarity=semantic_sim,
                token_similarity=token_sim,
                control_flow_similarity=control_flow_sim,
                data_flow_similarity=data_flow_sim,
                functional_similarity=functional_sim,
                confidence_interval=confidence_interval,
                p_value=p_value,
                statistical_significance=is_significant,
                evidence_strength=evidence_strength,
                obfuscation_detected=obfuscation_detected,
                transformation_patterns=transformations
            )
            
        except Exception as e:
            logger.warning(f"Failed to compare {file_a.path} and {file_b.path}: {e}")
            return None
    
    def _calculate_token_similarity(self, file_a: CodeFeatures, file_b: CodeFeatures) -> float:
        """Calculate token-based similarity"""
        tokens_a = file_a.normalized_code.split()
        tokens_b = file_b.normalized_code.split()
        return enhanced_jaccard_similarity(tokens_a, tokens_b)
    
    def _calculate_semantic_similarity(self, file_a: CodeFeatures, file_b: CodeFeatures) -> float:
        """Calculate semantic similarity using embeddings"""
        return cosine_similarity(file_a.semantic_embedding, file_b.semantic_embedding)
    
    def _calculate_structural_similarity(self, file_a: CodeFeatures, file_b: CodeFeatures) -> float:
        """Calculate structural similarity"""
        return structural_similarity(file_a.ast_structure, file_b.ast_structure)
    
    def _calculate_control_flow_similarity(self, file_a: CodeFeatures, file_b: CodeFeatures) -> float:
        """Calculate control flow similarity"""
        if not file_a.control_flow_patterns or not file_b.control_flow_patterns:
            return 0.0
        
        set_a = set(file_a.control_flow_patterns)
        set_b = set(file_b.control_flow_patterns)
        
        if not set_a and not set_b:
            return 1.0
        
        if not set_a or not set_b:
            return 0.0
        
        return len(set_a & set_b) / len(set_a | set_b)
    
    def _calculate_data_flow_similarity(self, file_a: CodeFeatures, file_b: CodeFeatures) -> float:
        """Calculate data flow similarity"""
        if not file_a.data_flow_patterns or not file_b.data_flow_patterns:
            return 0.0
        
        set_a = set(file_a.data_flow_patterns)
        set_b = set(file_b.data_flow_patterns)
        
        if not set_a and not set_b:
            return 1.0
        
        if not set_a or not set_b:
            return 0.0
        
        return len(set_a & set_b) / len(set_a | set_b)
    
    def _calculate_functional_similarity(self, file_a: CodeFeatures, file_b: CodeFeatures) -> float:
        """Calculate functional/algorithmic similarity independent of syntax"""
        
        # Combine multiple functional similarity measures
        similarities = []
        
        # 1. Control flow + data flow combined score
        cf_sim = self._calculate_control_flow_similarity(file_a, file_b)
        df_sim = self._calculate_data_flow_similarity(file_a, file_b)
        flow_similarity = (cf_sim + df_sim) / 2
        similarities.append(flow_similarity)
        
        # 2. Call sequence similarity (API usage patterns)
        if file_a.call_sequences and file_b.call_sequences:
            call_sim = self._calculate_sequence_similarity(file_a.call_sequences, file_b.call_sequences)
            similarities.append(call_sim)
        
        # 3. Complexity similarity (algorithmic complexity patterns)
        if file_a.complexity > 0 and file_b.complexity > 0:
            complexity_ratio = min(file_a.complexity, file_b.complexity) / max(file_a.complexity, file_b.complexity)
            similarities.append(complexity_ratio)
        
        # 4. Structural pattern similarity (functions, classes, variables ratios)
        structural_patterns = []
        for attr in ['functions', 'classes', 'variables']:
            val_a = getattr(file_a, attr, 0)
            val_b = getattr(file_b, attr, 0)
            if val_a > 0 and val_b > 0:
                ratio = min(val_a, val_b) / max(val_a, val_b)
                structural_patterns.append(ratio)
        
        if structural_patterns:
            similarities.append(np.mean(structural_patterns))
        
        # Return average of all functional similarities
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_sequence_similarity(self, seq_a: List[str], seq_b: List[str]) -> float:
        """Calculate similarity between call sequences using longest common subsequence"""
        if not seq_a or not seq_b:
            return 0.0
        
        # Use difflib for sequence matching
        matcher = difflib.SequenceMatcher(None, seq_a, seq_b)
        return matcher.ratio()
    
    def _establish_baseline(self, repo_a_files: List[CodeFeatures], repo_b_files: List[CodeFeatures]):
        """Establish statistical baseline from random file pairs"""
        import random
        
        # Filter out None values
        valid_files_a = [f for f in repo_a_files if f is not None]
        valid_files_b = [f for f in repo_b_files if f is not None]
        
        if not valid_files_a or not valid_files_b:
            logger.warning("No valid files for baseline establishment")
            return
        
        # Create random pairs for baseline (max 100 pairs to avoid excessive computation)
        all_files = valid_files_a + valid_files_b
        if len(all_files) < 2:
            logger.warning("Insufficient files for baseline establishment")
            return
            
        num_samples = min(100, len(all_files) * (len(all_files) - 1) // 2)
        
        random_pairs = []
        for _ in range(num_samples):
            file_a, file_b = random.sample(all_files, 2)
            random_pairs.append((file_a, file_b))
        
        self.statistical_analyzer.establish_baseline(random_pairs)
    
    def _determine_evidence_strength(self, similarity: float, clone_type: int, 
                                   is_significant: bool, transformations: List[str]) -> str:
        """Determine forensic evidence strength with cross-language considerations"""
        # Cross-language detection (Clone Type 4) requires semantic-focused thresholds
        if clone_type == 4:  # Cross-language semantic clones
            # For cross-language, prioritize semantic similarity over overall similarity
            # Use semantic similarity from the last computed match if available
            semantic_sim = getattr(self, '_last_semantic_similarity', similarity)
            
            # Strong evidence: High semantic similarity (>90%) OR high overall similarity (>65%)
            if semantic_sim > 0.90 or similarity > 0.65:
                return "STRONG"
            # Moderate evidence: Good semantic similarity (>85%) OR moderate overall (>55%)
            elif semantic_sim > 0.85 or similarity > 0.55:
                return "MODERATE"
            else:
                return "WEAK"
        
        # Traditional same-language detection
        if clone_type == 1 and similarity > 0.95:
            return "STRONG"
        elif clone_type <= 2 and similarity > 0.85 and is_significant:
            return "STRONG"
        elif clone_type <= 3 and similarity > 0.75 and is_significant:
            return "MODERATE"
        elif similarity > 0.65 and is_significant:
            return "MODERATE"
        else:
            return "WEAK"
    
    def _detect_obfuscation(self, file_a: CodeFeatures, file_b: CodeFeatures, 
                          transformations: List[str]) -> bool:
        """Detect potential obfuscation attempts"""
        obfuscation_indicators = [
            "identifier_renaming",
            "statement_addition",
            "statement_deletion",
            "control_flow_change",
            "function_modification"
        ]
        
        # Check for multiple transformation patterns
        detected_patterns = sum(1 for pattern in transformations if pattern in obfuscation_indicators)
        
        # Check for suspicious identifier patterns
        if file_a.identifier_map and file_b.identifier_map:
            # Look for systematic renaming patterns
            id_ratio_a = len(file_a.canonical_identifiers) / max(len(file_a.identifier_map), 1)
            id_ratio_b = len(file_b.canonical_identifiers) / max(len(file_b.identifier_map), 1)
            
            if abs(id_ratio_a - id_ratio_b) > 0.3:  # Significant difference in identifier patterns
                detected_patterns += 1
        
        return detected_patterns >= 2

# ----------------------------
# Enhanced Forensic Reporting
# ----------------------------
class ForensicReporter:
    """Generate comprehensive forensic reports with detailed analysis"""
    
    def __init__(self):
        self.report_timestamp = datetime.datetime.now().isoformat()
    
    def generate_comprehensive_report(self, matches: List[SimilarityResult], 
                                    repo_a_path: str, repo_b_path: str,
                                    analysis_params: Dict[str, Any],
                                    output_zip: str = 'evidence_package.zip'):
        """Generate complete forensic evidence package"""
        logger.info("Generating comprehensive forensic report...")
        
        # Generate all report formats
        csv_report = self._generate_csv_report(matches)
        html_report = self._generate_html_report(matches, repo_a_path, repo_b_path, analysis_params)
        json_report = self._generate_json_report(matches, repo_a_path, repo_b_path, analysis_params)
        executive_summary = self._generate_executive_summary(matches, analysis_params)
        technical_analysis = self._generate_technical_analysis(matches)
        
        # Create forensic evidence package
        self._create_evidence_package(
            matches, csv_report, html_report, json_report, 
            executive_summary, technical_analysis, output_zip
        )
        
        logger.info(f"Forensic evidence package created: {output_zip}")
    
    def _generate_csv_report(self, matches: List[SimilarityResult]) -> str:
        """Generate detailed CSV report"""
        if not matches:
            return ""
        
        csv_content = []
        fieldnames = [
            'file_a', 'file_b', 'clone_type', 'overall_similarity',
            'structural_similarity', 'semantic_similarity', 'token_similarity',
            'control_flow_similarity', 'data_flow_similarity', 'functional_similarity',
            'confidence_interval_lower', 'confidence_interval_upper',
            'p_value', 'statistical_significance', 'evidence_strength',
            'obfuscation_detected', 'transformation_patterns'
        ]
        
        csv_content.append(','.join(fieldnames))
        
        for match in matches:
            row = [
                match.file_a, match.file_b, str(match.clone_type),
                f"{match.overall_similarity:.4f}",
                f"{match.structural_similarity:.4f}",
                f"{match.semantic_similarity:.4f}",
                f"{match.token_similarity:.4f}",
                f"{match.control_flow_similarity:.4f}",
                f"{match.data_flow_similarity:.4f}",
                f"{match.functional_similarity:.4f}",
                f"{match.confidence_interval[0]:.4f}",
                f"{match.confidence_interval[1]:.4f}",
                f"{match.p_value:.6f}",
                str(match.statistical_significance),
                match.evidence_strength,
                str(match.obfuscation_detected),
                ';'.join(match.transformation_patterns)
            ]
            csv_content.append(','.join(f'"{item}"' for item in row))
        
        return '\n'.join(csv_content)
    
    def _generate_html_report(self, matches: List[SimilarityResult], 
                            repo_a_path: str, repo_b_path: str,
                            analysis_params: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            "<title>SIPCompare Forensic Analysis Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; }",
            ".header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }",
            ".match { border: 1px solid #ccc; margin: 20px 0; padding: 15px; border-radius: 5px; }",
            ".strong { background-color: #ffebee; }",
            ".moderate { background-color: #fff3e0; }",
            ".weak { background-color: #f3e5f5; }",
            ".metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 10px 0; }",
            ".metric { background-color: #f5f5f5; padding: 10px; border-radius: 3px; text-align: center; }",
            ".diff { background-color: #f8f8f8; padding: 10px; font-family: monospace; white-space: pre-wrap; }",
            "</style>",
            "</head><body>"
        ]
        
        # Header section
        html_parts.extend([
            "<div class='header'>",
            "<h1>SIPCompare Forensic Analysis Report</h1>",
            f"<p><strong>Analysis Date:</strong> {self.report_timestamp}</p>",
            f"<p><strong>Repository A:</strong> {repo_a_path}</p>",
            f"<p><strong>Repository B:</strong> {repo_b_path}</p>",
            f"<p><strong>Threshold:</strong> {analysis_params.get('threshold', 'N/A')}</p>",
            f"<p><strong>Embedding Model:</strong> {analysis_params.get('embedding_model', 'N/A')}</p>",
            f"<p><strong>Total Matches:</strong> {len(matches)}</p>",
            "</div>"
        ])
        
        # Summary statistics
        if matches:
            strong_matches = sum(1 for m in matches if m.evidence_strength == "STRONG")
            moderate_matches = sum(1 for m in matches if m.evidence_strength == "MODERATE")
            weak_matches = sum(1 for m in matches if m.evidence_strength == "WEAK")
            obfuscation_detected = sum(1 for m in matches if m.obfuscation_detected)
            
            html_parts.extend([
                "<h2>Summary Statistics</h2>",
                "<div class='metrics'>",
                f"<div class='metric'><strong>Strong Evidence:</strong><br>{strong_matches}</div>",
                f"<div class='metric'><strong>Moderate Evidence:</strong><br>{moderate_matches}</div>",
                f"<div class='metric'><strong>Weak Evidence:</strong><br>{weak_matches}</div>",
                f"<div class='metric'><strong>Obfuscation Detected:</strong><br>{obfuscation_detected}</div>",
                f"<div class='metric'><strong>Avg Similarity:</strong><br>{np.mean([m.overall_similarity for m in matches]):.3f}</div>",
                f"<div class='metric'><strong>Statistical Significance:</strong><br>{sum(1 for m in matches if m.statistical_significance)}</div>",
                "</div>"
            ])
        
        # Individual matches
        html_parts.append("<h2>Detailed Match Analysis</h2>")
        
        for i, match in enumerate(matches, 1):
            css_class = match.evidence_strength.lower()
            
            html_parts.extend([
                f"<div class='match {css_class}'>",
                f"<h3>Match #{i} - {match.evidence_strength} Evidence</h3>",
                f"<p><strong>File A:</strong> {match.file_a}</p>",
                f"<p><strong>File B:</strong> {match.file_b}</p>",
                f"<p><strong>Clone Type:</strong> Type {match.clone_type}</p>",
                f"<p><strong>Overall Similarity:</strong> {match.overall_similarity:.4f}</p>",
                f"<p><strong>Statistical Significance:</strong> {'Yes' if match.statistical_significance else 'No'} (p={match.p_value:.6f})</p>",
                f"<p><strong>Confidence Interval:</strong> [{match.confidence_interval[0]:.4f}, {match.confidence_interval[1]:.4f}]</p>",
                f"<p><strong>Obfuscation Detected:</strong> {'Yes' if match.obfuscation_detected else 'No'}</p>",
                f"<p><strong>Transformations:</strong> {', '.join(match.transformation_patterns) if match.transformation_patterns else 'None'}</p>",
                
                "<div class='metrics'>",
                f"<div class='metric'><strong>Token Similarity:</strong><br>{match.token_similarity:.4f}</div>",
                f"<div class='metric'><strong>Semantic Similarity:</strong><br>{match.semantic_similarity:.4f}</div>",
                f"<div class='metric'><strong>Structural Similarity:</strong><br>{match.structural_similarity:.4f}</div>",
                f"<div class='metric'><strong>Control Flow:</strong><br>{match.control_flow_similarity:.4f}</div>",
                f"<div class='metric'><strong>Data Flow:</strong><br>{match.data_flow_similarity:.4f}</div>",
                f"<div class='metric'><strong>Functional:</strong><br>{match.functional_similarity:.4f}</div>",
                "</div>"
            ])
            
            # Add code diff if files exist
            try:
                with open(match.file_a, 'r', errors='ignore') as fa:
                    with open(match.file_b, 'r', errors='ignore') as fb:
                        diff = list(difflib.unified_diff(
                            fa.readlines(), fb.readlines(),
                            fromfile=f"A: {os.path.basename(match.file_a)}",
                            tofile=f"B: {os.path.basename(match.file_b)}",
                            n=3
                        ))
                        
                        if diff:
                            html_parts.extend([
                                "<h4>Code Diff:</h4>",
                                "<div class='diff'>",
                                ''.join(diff[:50]),  # Limit diff size
                                "</div>"
                            ])
            except Exception:
                html_parts.append("<p><em>Code diff unavailable</em></p>")
            
            html_parts.append("</div>")
        
        html_parts.extend(["</body></html>"])
        
        return '\n'.join(html_parts)
    
    def _generate_json_report(self, matches: List[SimilarityResult], 
                            repo_a_path: str, repo_b_path: str,
                            analysis_params: Dict[str, Any]) -> str:
        """Generate detailed JSON report"""
        report_data = {
            "metadata": {
                "tool": "SIPCompare",
                "version": "2.0",
                "timestamp": self.report_timestamp,
                "repository_a": repo_a_path,
                "repository_b": repo_b_path,
                "analysis_parameters": analysis_params,
                "total_matches": len(matches)
            },
            "summary": {
                "strong_evidence": sum(1 for m in matches if m.evidence_strength == "STRONG"),
                "moderate_evidence": sum(1 for m in matches if m.evidence_strength == "MODERATE"),
                "weak_evidence": sum(1 for m in matches if m.evidence_strength == "WEAK"),
                "obfuscation_detected": sum(1 for m in matches if m.obfuscation_detected),
                "statistically_significant": sum(1 for m in matches if m.statistical_significance),
                "average_similarity": np.mean([m.overall_similarity for m in matches]) if matches else 0.0
            },
            "matches": [asdict(match) for match in matches]
        }
        
        return json.dumps(report_data, indent=2, default=str)
    
    def _generate_executive_summary(self, matches: List[SimilarityResult], 
                                  analysis_params: Dict[str, Any]) -> str:
        """Generate executive summary for non-technical stakeholders"""
        
        # Simple conclusion statement
        if not matches:
            conclusion = "CONCLUSION: The analysis of the compared code indicates that files were NOT copied and/or transformed."
        else:
            strong_matches = sum(1 for m in matches if m.evidence_strength == "STRONG")
            moderate_matches = sum(1 for m in matches if m.evidence_strength == "MODERATE")
            obfuscation_count = sum(1 for m in matches if m.obfuscation_detected)
            
            if strong_matches > 0 or (moderate_matches > 0 and obfuscation_count > 0):
                conclusion = "CONCLUSION: The analysis of the compared code indicates that files WERE copied and/or transformed."
            elif moderate_matches > 0:
                conclusion = "CONCLUSION: The analysis suggests files MAY HAVE BEEN copied and/or transformed (requires further investigation)."
            else:
                conclusion = "CONCLUSION: The analysis of the compared code indicates that files were NOT copied and/or transformed."
        
        if not matches:
            return f"{conclusion}\n\nNo significant code similarities detected between the repositories."
        
        strong_matches = sum(1 for m in matches if m.evidence_strength == "STRONG")
        moderate_matches = sum(1 for m in matches if m.evidence_strength == "MODERATE")
        weak_matches = sum(1 for m in matches if m.evidence_strength == "WEAK")
        obfuscation_count = sum(1 for m in matches if m.obfuscation_detected)
        
        summary_parts = [
            "EXECUTIVE SUMMARY",
            "=" * 50,
            "",
            conclusion,
            "",
            f"Analysis completed on {self.report_timestamp}",
            f"Total file pairs with similarities: {len(matches)}",
            "",
            "EVIDENCE SUMMARY:",
            f"• Strong evidence (high confidence): {strong_matches} cases",
            f"• Moderate evidence (medium confidence): {moderate_matches} cases", 
            f"• Weak evidence (low confidence): {weak_matches} cases",
            f"• Potential obfuscation attempts detected: {obfuscation_count} cases",
            "",
            "RISK ASSESSMENT:",
        ]
        
        if strong_matches > 0:
            summary_parts.append("• HIGH RISK: Strong evidence suggests potential intellectual property theft")
            summary_parts.append("• RECOMMENDATION: Legal consultation strongly recommended")
        elif moderate_matches > 0:
            summary_parts.append("• MEDIUM RISK: Moderate evidence requires further investigation")
            summary_parts.append("• RECOMMENDATION: Detailed review and possible legal consultation")
        else:
            summary_parts.append("• LOW RISK: Only weak similarities detected")
            summary_parts.append("• RECOMMENDATION: Monitor for patterns, no immediate action required")
        
        if obfuscation_count > 0:
            summary_parts.extend([
                "",
                "OBFUSCATION DETECTED:",
                "• Evidence of deliberate code modification to hide similarities",
                "• This strengthens the case for intentional copying/transformation",
                "• Suggests awareness of wrongdoing and attempt to conceal it"
            ])
        
        summary_parts.extend([
            "",
            "NEXT STEPS:",
            "• Review detailed technical analysis for specific file comparisons",
            "• Examine evidence files included in this package",
            "• Consider forensic chain of custody documentation",
            "• Implement code review processes to prevent future incidents",
            ""
        ])
        
        return '\n'.join(summary_parts)
    
    def _generate_technical_analysis(self, matches: List[SimilarityResult]) -> str:
        """Generate detailed technical analysis"""
        if not matches:
            return "No matches found for technical analysis."
        
        analysis_parts = [
            "TECHNICAL ANALYSIS REPORT",
            "=" * 50,
            "",
            "METHODOLOGY:",
            "• Multi-dimensional similarity analysis using:",
            "  - Token-based comparison (Jaccard similarity with weighting)",
            "  - Semantic analysis (transformer-based embeddings)",
            "  - Structural analysis (AST-based features)",
            "  - Control flow pattern matching",
            "  - Statistical significance testing",
            "",
            "CLONE TYPE DISTRIBUTION:",
        ]
        
        clone_types = Counter(m.clone_type for m in matches)
        for clone_type, count in sorted(clone_types.items()):
            type_desc = {
                1: "Exact clones (whitespace/comment differences only)",
                2: "Renamed clones (identifier changes)",
                3: "Near-miss clones (statement modifications)",
                4: "Semantic clones (different syntax, same functionality)"
            }
            analysis_parts.append(f"• Type {clone_type}: {count} cases - {type_desc.get(clone_type, 'Unknown')}")
        
        analysis_parts.extend([
            "",
            "TRANSFORMATION PATTERNS DETECTED:",
        ])
        
        all_transformations = []
        for match in matches:
            all_transformations.extend(match.transformation_patterns)
        
        transformation_counts = Counter(all_transformations)
        for transformation, count in transformation_counts.most_common():
            analysis_parts.append(f"• {transformation}: {count} occurrences")
        
        analysis_parts.extend([
            "",
            "STATISTICAL ANALYSIS:",
            f"• Mean similarity score: {np.mean([m.overall_similarity for m in matches]):.4f}",
            f"• Standard deviation: {np.std([m.overall_similarity for m in matches]):.4f}",
            f"• Statistically significant matches: {sum(1 for m in matches if m.statistical_significance)}",
            f"• Average p-value: {np.mean([m.p_value for m in matches]):.6f}",
            ""
        ])
        
        return '\n'.join(analysis_parts)
    
    def _create_evidence_package(self, matches: List[SimilarityResult],
                               csv_report: str, html_report: str, json_report: str,
                               executive_summary: str, technical_analysis: str,
                               output_zip: str):
        """Create comprehensive forensic evidence package"""
        with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # Add reports
            zf.writestr('reports/detailed_analysis.csv', csv_report)
            zf.writestr('reports/forensic_report.html', html_report)
            zf.writestr('reports/analysis_data.json', json_report)
            zf.writestr('reports/executive_summary.txt', executive_summary)
            zf.writestr('reports/technical_analysis.txt', technical_analysis)
            
            # Add source files for evidence
            added_files = set()
            for match in matches:
                for file_path in [match.file_a, match.file_b]:
                    if file_path not in added_files:
                        try:
                            arcname = f"evidence_files/{os.path.relpath(file_path)}"
                            zf.write(file_path, arcname=arcname)
                            added_files.add(file_path)
                        except Exception as e:
                            logger.warning(f"Could not add {file_path} to evidence package: {e}")
            
            # Add chain of custody document
            custody_doc = self._generate_chain_of_custody(matches)
            zf.writestr('chain_of_custody.txt', custody_doc)
    
    def _generate_chain_of_custody(self, matches: List[SimilarityResult]) -> str:
        """Generate chain of custody documentation"""
        custody_parts = [
            "CHAIN OF CUSTODY DOCUMENTATION",
            "=" * 50,
            "",
            f"Analysis Tool: SIPCompare v2.0",
            f"Analysis Date: {self.report_timestamp}",
            f"Total Evidence Files: {len(set([m.file_a for m in matches] + [m.file_b for m in matches]))}",
            f"Total Matches: {len(matches)}",
            "",
            "EVIDENCE INTEGRITY:",
            "• All source files included in evidence package",
            "• SHA256 hashes computed for file integrity verification",
            "• Statistical analysis performed with confidence intervals",
            "• Multiple similarity algorithms applied for validation",
            "",
            "ANALYSIS PARAMETERS:",
            "• Advanced AST-based structural analysis",
            "• Semantic embedding comparison using transformer models",
            "• Obfuscation detection algorithms",
            "• Statistical significance testing",
            "",
            "EVIDENCE FILES:",
        ]
        
        evidence_files = set()
        for match in matches:
            evidence_files.add(match.file_a)
            evidence_files.add(match.file_b)
        
        for i, file_path in enumerate(sorted(evidence_files), 1):
            custody_parts.append(f"{i:3d}. {file_path}")
        
        custody_parts.extend([
            "",
            "This evidence package maintains forensic integrity and can be used",
            "for legal proceedings or intellectual property investigations.",
            ""
        ])
        
        return '\n'.join(custody_parts)

# ----------------------------
# Main Analysis Engine
# ----------------------------
def analyze_repositories(repo_a_path: str, repo_b_path: str, 
                        threshold: float = 0.75, 
                        embedding_model: str = 'graphcodebert',
                        parallel_workers: int = 1,
                        enable_statistical: bool = True,
                        cross_language: bool = False,
                        output_zip: str = 'evidence_package.zip') -> List[SimilarityResult]:
    """
    Main analysis function with enhanced capabilities
    
    Args:
        repo_a_path: Path to first repository
        repo_b_path: Path to second repository
        threshold: Similarity threshold (0-1)
        embedding_model: Model to use ('mini', 'graphcodebert', 'codet5')
        parallel_workers: Number of parallel workers
        enable_statistical: Enable statistical analysis
        output_zip: Output evidence package path
    
    Returns:
        List of similarity results
    """
    logger.info("Starting SIPCompare v2.0 analysis...")
    logger.info(f"Repository A: {repo_a_path}")
    logger.info(f"Repository B: {repo_b_path}")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Embedding Model: {embedding_model}")
    logger.info(f"Parallel Workers: {parallel_workers}")
    
    # Collect and process files
    logger.info("Phase 1: Collecting and processing files...")
    repo_a_files = collect_files_parallel(repo_a_path, embedding_model, parallel_workers)
    repo_b_files = collect_files_parallel(repo_b_path, embedding_model, parallel_workers)
    
    if not repo_a_files:
        logger.error(f"No processable files found in {repo_a_path}")
        return []
    
    if not repo_b_files:
        logger.error(f"No processable files found in {repo_b_path}")
        return []
    
    logger.info(f"Processed {len(repo_a_files)} files from Repository A")
    logger.info(f"Processed {len(repo_b_files)} files from Repository B")
    
    # Perform similarity analysis
    logger.info("Phase 2: Performing similarity analysis...")
    similarity_engine = SimilarityEngine(statistical_analysis=enable_statistical, cross_language=cross_language)
    matches = similarity_engine.compare_repositories(
        repo_a_files, repo_b_files, threshold, enable_statistical
    )
    
    if not matches:
        logger.info("No matches found above the specified threshold")
        return []
    
    # Generate forensic report
    logger.info("Phase 3: Generating forensic evidence package...")
    analysis_params = {
        'threshold': threshold,
        'embedding_model': embedding_model,
        'parallel_workers': parallel_workers,
        'statistical_analysis': enable_statistical,
        'tool_version': '2.0'
    }
    
    reporter = ForensicReporter()
    reporter.generate_comprehensive_report(
        matches, repo_a_path, repo_b_path, analysis_params, output_zip
    )
    
    # Summary statistics
    strong_matches = sum(1 for m in matches if m.evidence_strength == "STRONG")
    moderate_matches = sum(1 for m in matches if m.evidence_strength == "MODERATE")
    obfuscation_detected = sum(1 for m in matches if m.obfuscation_detected)
    
    logger.info("=" * 60)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total matches found: {len(matches)}")
    logger.info(f"Strong evidence: {strong_matches}")
    logger.info(f"Moderate evidence: {moderate_matches}")
    logger.info(f"Obfuscation detected: {obfuscation_detected}")
    logger.info(f"Evidence package: {output_zip}")
    logger.info("=" * 60)
    
    return matches

# ----------------------------
# Command Line Interface
# ----------------------------
def main():
    """Enhanced command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SIPCompare v2.0: Advanced Forensic Multi-Language Semantic Code Similarity Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python SIPCompare.py --repoA /path/to/repo1 --repoB /path/to/repo2
  
  # High-sensitivity analysis with parallel processing
  python SIPCompare.py --repoA /path/to/repo1 --repoB /path/to/repo2 \\
                       --threshold 0.6 --parallel 4 --embedding-model graphcodebert
  
  # Cross-language analysis
  python SIPCompare.py --repoA /path/to/python_repo --repoB /path/to/java_repo \\
                       --cross-language --embedding-model codet5
        """
    )
    
    # Required arguments
    parser.add_argument("--repoA", required=True, 
                       help="Path to first repository")
    parser.add_argument("--repoB", required=True, 
                       help="Path to second repository")
    
    # Analysis parameters
    parser.add_argument("--threshold", type=float, default=0.50,
                       help="Similarity threshold (0-1), default: 0.50 (lowered for better cross-language detection)")
    parser.add_argument("--embedding-model", type=str, default='graphcodebert',
                       choices=['mini', 'graphcodebert', 'codet5'],
                       help="Embedding model to use, default: graphcodebert")
    
    # Performance options
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of parallel processes, default: 1")
    
    # Analysis options
    parser.add_argument("--no-statistical", action='store_true',
                       help="Disable statistical significance testing")
    parser.add_argument("--cross-language", action='store_true',
                       help="Enable cross-language detection (experimental)")
    
    # Output options
    parser.add_argument("--output", type=str, default='evidence_package.zip',
                       help="Output forensic evidence package, default: evidence_package.zip")
    parser.add_argument("--verbose", action='store_true',
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.repoA):
        logger.error(f"Repository A path does not exist: {args.repoA}")
        return 1
    
    if not os.path.exists(args.repoB):
        logger.error(f"Repository B path does not exist: {args.repoB}")
        return 1
    
    if not (0.0 <= args.threshold <= 1.0):
        logger.error(f"Threshold must be between 0.0 and 1.0, got: {args.threshold}")
        return 1
    
    if args.parallel < 1:
        logger.error(f"Parallel workers must be >= 1, got: {args.parallel}")
        return 1
    
    # Adjust parallel workers based on system capabilities
    max_workers = min(args.parallel, mp.cpu_count())
    if max_workers != args.parallel:
        logger.warning(f"Reducing parallel workers from {args.parallel} to {max_workers} (system limit)")
    
    try:
        # Run analysis
        matches = analyze_repositories(
            repo_a_path=args.repoA,
            repo_b_path=args.repoB,
            threshold=args.threshold,
            embedding_model=args.embedding_model,
            parallel_workers=max_workers,
            enable_statistical=not args.no_statistical,
            cross_language=args.cross_language,
            output_zip=args.output
        )
        
        # Exit with appropriate code
        if matches:
            strong_matches = sum(1 for m in matches if m.evidence_strength == "STRONG")
            if strong_matches > 0:
                logger.warning("Strong evidence of code similarity detected!")
                return 2  # Strong evidence exit code
            else:
                return 0  # Normal completion
        else:
            return 0  # No matches found
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
