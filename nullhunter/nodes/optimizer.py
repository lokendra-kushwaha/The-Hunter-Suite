"""
NullHunter Schema Optimizer (The Semantic Engine)
=============================================================

Enterprise-Grade Semantic Schema Optimizer.
Operates via a Dual-API Architecture to enforce ML-ready and SQL-safe column headers.

- Micro-API (`optimize_metadata`): Designed for the NullHunter Engine. Executes O(1) 
  transformations purely on string lists during Pass 1 to generate 'Asset 2' and 
  the Translation Ledger.
- Macro-API (`optimize_dataframe`): Standalone entry point for independent Data Scientists 
  applying transformations directly to Pandas DataFrames.
"""

import re
import logging
import difflib
import unicodedata
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from collections import Counter

# ==========================================
# CUSTOM EXCEPTIONS
# ==========================================
class NullHunterSchemaError(Exception):
    """Raised when there is an invalid schema configuration or structural mismatch."""
    pass

# ==========================================
# LOGGER CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] NULLHUNTER_OPTIMIZER: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SchemaOptimizer:
    """
    Advanced Semantic Schema Standardizer.

    Attributes:
        case_style (str): The target casing format ('snake', 'camel', 'pascal', 'upper_snake').
        resolve_duplicates (bool): Appends numerical suffixes to prevent Pandas collision.
        sql_safe_mode (bool): Renames columns matching reserved SQL keywords.
        max_col_length (int): Truncates columns to prevent DB insertion crashes (Default 63).
        normalize_unicode (bool): Converts international characters to ASCII (e.g., 'é' -> 'e').
        regex_replacements (Dict[str, str]): Symbol-to-word translation mapping.
        ai_auto_correct (bool): Uses fuzzy string matching to fix semantic typos.
        auto_correct_threshold (float): Similarity threshold (0.0 to 1.0) for AI matching.
        standard_dictionary (List[str]): Valid standard column names for AI matching.
        custom_columns_map (Dict[str, str]): Hardcoded overrides {old: new}.
    """

    # Global standard list of reserved SQL/Pandas keywords
    RESERVED_KEYWORDS = {
        'select', 'insert', 'update', 'delete', 'drop', 'table', 'index', 
        'where', 'join', 'sum', 'count', 'group', 'by', 'having', 'limit'
    }

    def __init__(
        self,
        case_style: str = "snake",
        resolve_duplicates: bool = True,
        sql_safe_mode: bool = True,
        max_col_length: int = 63,
        normalize_unicode: bool = True,
        custom_regex_replacements: Optional[Dict[str, str]] = None,
        ai_auto_correct: bool = False,
        auto_correct_threshold: float = 0.80,
        domain_dictionary: Optional[List[str]] = None,
        custom_columns_map: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initializes the SchemaOptimizer with enterprise formatting parameters.

        Args:
            case_style (str): Casing format. Options: 'snake', 'camel', 'pascal', 'upper_snake'.
            resolve_duplicates (bool): Appends suffixes to duplicates (e.g., sales_1).
            sql_safe_mode (bool): Modifies reserved SQL keywords (e.g., 'select' -> 'select_col').
            max_col_length (int): Maximum character length for headers.
            normalize_unicode (bool): Strips accents and normalizes international text.
            custom_regex_replacements (Optional[Dict[str, str]]): Pre-cleaning string translation.
            ai_auto_correct (bool): Fuzzy matching for typo resolution.
            auto_correct_threshold (float): Strictness of the AI typo fixer.
            domain_dictionary (Optional[List[str]]): Target dictionary for AI fuzzy matching.
            custom_columns_map (Optional[Dict[str, str]]): Explicit overrides mapping.
        """
        self.case_style = case_style.lower()
        self.resolve_duplicates = resolve_duplicates
        self.sql_safe_mode = sql_safe_mode
        self.max_col_length = max_col_length
        self.normalize_unicode = normalize_unicode
        self.ai_auto_correct = ai_auto_correct
        self.auto_correct_threshold = max(0.1, min(auto_correct_threshold, 1.0))
        
        self.regex_replacements = custom_regex_replacements or {
            '%': 'percent', '#': 'num', '$': 'usd', '&': 'and', '@': 'at', '+': 'plus'
        }
        self.custom_columns_map = custom_columns_map or {}
        
        self.standard_dictionary = domain_dictionary or [
            'id', 'first_name', 'last_name', 'age', 'gender', 'email', 'phone', 'address', 
            'city', 'state', 'zipcode', 'country', 'salary', 'revenue', 'profit', 'date', 
            'date_of_birth', 'status', 'category', 'amount', 'quantity', 'price', 'latitude', 'longitude'
        ]
        
        if self.case_style not in ['snake', 'camel', 'pascal', 'upper_snake']:
            raise ValueError(f"Invalid case_style '{self.case_style}'. Use snake, camel, pascal, or upper_snake.")
            
        logger.debug(
            f"SchemaOptimizer Armed (Style: {self.case_style.upper()} | "
            f"SQL-Safe: {self.sql_safe_mode} | Max Len: {self.max_col_length})."
        )

    def _apply_case_style(self, text: str) -> str:
        """Applies the configured casing convention to a cleaned alphanumeric string."""
        if not text:
            return text
            
        words = text.split('_')
        
        if self.case_style == 'snake':
            return text.lower()
        elif self.case_style == 'upper_snake':
            return text.upper()
        elif self.case_style == 'camel':
            return words[0].lower() + ''.join(word.capitalize() for word in words[1:])
        elif self.case_style == 'pascal':
            return ''.join(word.capitalize() for word in words)
        return text

    def _clean_column_string(self, col_name: Any) -> str:
        """
        Master string cleaning pipeline. Translates special chars, standardizes unicode,
        removes illegal characters, enforces length, and applies database casing.
        """
        # Type-Safe Cast (Protects against datasets with integer headers)
        raw_str = str(col_name).strip() if pd.notna(col_name) else ""
        if not raw_str:
            return "unnamed_col"

        # 1. Unicode Normalization (e.g., 'Café' -> 'Cafe')
        if self.normalize_unicode:
            raw_str = unicodedata.normalize('NFKD', raw_str).encode('ascii', 'ignore').decode('utf-8')

        # 2. Context Preservation
        for symbol, word in self.regex_replacements.items():
            raw_str = raw_str.replace(symbol, f"_{word}_")

        # 3. Strip non-alphanumeric and convert to intermediate snake_case
        cleaned = re.sub(r'[^a-zA-Z0-9]', '_', raw_str.lower())
        
        # 4. Collapse multiple underscores and strip borders
        cleaned = re.sub(r'_+', '_', cleaned).strip('_')
        
        if not cleaned:
            return "unnamed_col"

        # 5. SQL Safe Mode (Prevent Database Crashes)
        if self.sql_safe_mode and cleaned in self.RESERVED_KEYWORDS:
            cleaned = f"{cleaned}_col"

        # 6. Enforce Database Length Constraints
        cleaned = cleaned[:self.max_col_length].strip('_')

        # 7. Apply Final Casing
        return self._apply_case_style(cleaned)

    def _apply_ai_auto_correct(self, col_name: str) -> str:
        """Fuzzy matches column headers to the Domain Dictionary."""
        # Normalize casing for accurate comparison
        search_target = col_name.lower()
        dictionary_lower = [term.lower() for term in self.standard_dictionary]
        
        matches = difflib.get_close_matches(
            search_target, dictionary_lower, n=1, cutoff=self.auto_correct_threshold
        )
        if matches:
            # Re-apply the configured case style to the matched word
            return self._apply_case_style(matches[0])
        return col_name

    def _resolve_collisions(self, cols: List[str]) -> List[str]:
        """Ensures absolute uniqueness in the column array to prevent Matrix crashes."""
        counts = Counter(cols)
        if not any(count > 1 for count in counts.values()):
            return cols

        seen = {}
        resolved = []
        for name in cols:
            if counts[name] > 1:
                seen[name] = seen.get(name, 0) + 1
                suffix = f"_{seen[name]}"
                if self.case_style in ['camel', 'pascal']:
                    suffix = str(seen[name])
                
                new_name = f"{name}{suffix}"
                # Ensure suffix addition didn't violate SQL length limits
                if len(new_name) > self.max_col_length:
                    truncated_base = name[:self.max_col_length - len(suffix)]
                    new_name = f"{truncated_base}{suffix}"
                    
                resolved.append(new_name)
            else:
                resolved.append(name)
                
        logger.debug(f"Resolved {len([c for c in counts.values() if c > 1])} schema collisions.")
        return resolved

    def optimize_metadata(self, raw_columns: List[Any]) -> Tuple[List[str], Dict[str, str]]:
        """
        THE MICRO-API (FOR NULLHUNTER ENGINE - PASS 1)
        
        Processes purely the metadata (List of Strings/Ints) to avoid DataFrame overhead.
        Executed strictly once on Chunk 1 to generate 'Asset 2'.
        
        Args:
            raw_columns (List[Any]): The raw 2D columns outputted by the Flattener.
            
        Returns:
            Tuple[List[str], Dict[str, str]]:
                - Asset 2: The absolute finalized, optimized column array.
                - Translation Ledger: Mapping of {new_optimized_name : original_raw_name}.
        """
        logger.info("Generating Asset 2: Optimized Metadata Blueprint...")
        
        new_cols = []
        for col in raw_columns:
            # 1. Custom Override Check
            if col in self.custom_columns_map:
                new_cols.append(self.custom_columns_map[col])
                continue
                
            # 2. String Sanitization & Casing
            cleaned = self._clean_column_string(col)
            
            # 3. AI Typo Fixer
            if self.ai_auto_correct:
                cleaned = self._apply_ai_auto_correct(cleaned)
                
            new_cols.append(cleaned)
            
        # 4. Global Collision Resolution
        if self.resolve_duplicates:
            new_cols = self._resolve_collisions(new_cols)
            
        # 5. Ledger Generation (Strict mapping back to the EXACT original input)
        translation_ledger = {new: str(old) for new, old in zip(new_cols, raw_columns)}
        
        logger.info("Asset 2 Generated. Ready for deployment.")
        return new_cols, translation_ledger

    def optimize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        THE MACRO-API (FOR STANDALONE SAAS USAGE)
        
        Allows independent application of semantic optimizations on Pandas DataFrames.
        Includes structural safeguards against 3D hierarchies.
        
        Args:
            df (pd.DataFrame): The target DataFrame.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: The mutated DataFrame and Translation Ledger.
            
        Raises:
            NullHunterSchemaError: If the DataFrame contains a MultiIndex.
        """
        if isinstance(df.columns, pd.MultiIndex):
            raise NullHunterSchemaError(
                "Cannot semantically optimize a 3D MultiIndex DataFrame. "
                "Route the data through SchemaFlattener (Layer 1.5) before optimization."
            )
            
        logger.info("Standalone Mode: Optimizing DataFrame schema inplace...")
        
        raw_cols = list(df.columns)
        optimized_headers, ledger = self.optimize_metadata(raw_cols)
        
        optimized_df = df.copy(deep=False)
        optimized_df.columns = optimized_headers
        
        return optimized_df, ledger

    