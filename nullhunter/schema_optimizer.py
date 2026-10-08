import re
import logging
import difflib
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union, List
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
    Enterprise-Grade Semantic Schema Optimizer (Layer 1.8).
    
    Operates via a Dual-API Architecture:
    1. Micro-API (`optimize_metadata`): For the NullHunter Core Engine. Computes O(1) 
       transformations purely on string lists (headers) during Pass 1, Chunk 1. 
       Generates 'Asset 2' (Optimized Metadata) and a Translation Ledger.
    2. Macro-API (`optimize_dataframe`): For independent users who want to use this 
       tool directly on standard Pandas DataFrames outside the NullHunter pipeline.
    """

    # Global standard list of reserved SQL/Pandas keywords to prevent downstream database crashes
    RESERVED_KEYWORDS = {'select', 'insert', 'update', 'delete', 'drop', 'table', 'index', 'where', 'join', 'sum', 'count'}

    def __init__(
        self,
        case_style: str = "snake",
        resolve_duplicates: bool = True,
        sql_safe_mode: bool = True,
        custom_regex_replacements: Optional[Dict[str, str]] = None,
        ai_auto_correct: bool = False,
        auto_correct_threshold: float = 0.80,
        domain_dictionary: Optional[List[str]] = None,
        custom_columns_map: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initializes the SchemaOptimizer with highly advanced structural styling parameters.
        
        Args:
            case_style (str): The target casing format. Options: 'snake', 'camel', 'pascal', 'upper_snake'.
            resolve_duplicates (bool): Appends numerical suffixes to prevent Pandas column collision (e.g., sales_1).
            sql_safe_mode (bool): Renames columns that match reserved SQL keywords (e.g., 'select' -> 'select_col').
            custom_regex_replacements (Optional[Dict[str, str]]): Dictionary of string replacements before cleaning 
                                                                  (e.g., {'%': 'percent', '#': 'num'}).
            ai_auto_correct (bool): Uses fuzzy string matching to fix common semantic typos (e.g., 'gendre' -> 'gender').
            auto_correct_threshold (float): Similarity threshold (0.0 to 1.0) for AI fuzzy matching.
            domain_dictionary (Optional[List[str]]): A custom list of valid standard column names for AI matching.
            custom_columns_map (Optional[Dict[str, str]]): Hardcoded mapping to override specific column names {old: new}.
        """
        self.case_style = case_style.lower()
        self.resolve_duplicates = resolve_duplicates
        self.sql_safe_mode = sql_safe_mode
        self.ai_auto_correct = ai_auto_correct
        self.auto_correct_threshold = auto_correct_threshold
        
        # Default smart replacements to preserve context before special characters are stripped
        self.regex_replacements = custom_regex_replacements or {
            '%': 'percent', '#': 'num', '$': 'usd', '&': 'and', '@': 'at'
        }
        self.custom_columns_map = custom_columns_map or {}
        
        self.standard_dictionary = domain_dictionary or [
            'id', 'first_name', 'last_name', 'age', 'gender', 'email', 'phone', 'address', 
            'city', 'state', 'zipcode', 'country', 'salary', 'revenue', 'profit', 'date', 
            'date_of_birth', 'status', 'category', 'amount', 'quantity', 'price', 'latitude', 'longitude'
        ]
        
        if self.case_style not in ['snake', 'camel', 'pascal', 'upper_snake']:
            raise ValueError(f"Invalid case_style '{self.case_style}'. Use snake, camel, pascal, or upper_snake.")
            
        logger.info(f"SchemaOptimizer Armed (Style: {self.case_style.upper()} | SQL-Safe: {self.sql_safe_mode}).")


    def _apply_case_style(self, text: str) -> str:
        """Applies the selected casing convention to a cleaned alphanumeric string."""
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


    def _clean_column_string(self, col_name: str) -> str:
        """
        Master string cleaning pipeline. Translates special chars, strips whitespaces, 
        removes illegal characters, and applies database casing.
        """
        raw_str = str(col_name).strip()
        if pd.isna(col_name) or not raw_str:
            return "unnamed_col"

        # 1. Context Preservation (Replace specific symbols with words)
        for symbol, word in self.regex_replacements.items():
            raw_str = raw_str.replace(symbol, f"_{word}_")

        # 2. Strip non-alphanumeric and convert to intermediate snake_case
        cleaned = re.sub(r'[^a-zA-Z0-9]', '_', raw_str.lower())
        
        # 3. Collapse multiple underscores and strip trailing/leading
        cleaned = re.sub(r'_+', '_', cleaned).strip('_')
        
        if not cleaned:
            return "unnamed_col"

        # 4. SQL Safe Mode (Prevent Database Crashes)
        if self.sql_safe_mode and cleaned in self.RESERVED_KEYWORDS:
            cleaned = f"{cleaned}_col"

        # 5. Apply Final User Casing Preference
        return self._apply_case_style(cleaned)


    def _apply_ai_auto_correct(self, col_name: str) -> str:
        """Fuzzy matches broken column headers to the Domain Dictionary."""
        matches = difflib.get_close_matches(
            col_name, self.standard_dictionary, n=1, cutoff=self.auto_correct_threshold
        )
        if matches:
            return matches[0]
        return col_name


    def _resolve_collisions(self, cols: List[str]) -> List[str]:
        """Ensures absolute uniqueness in the column array to prevent downstream Matrix crashes."""
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
                    suffix = str(seen[name]) # Append number directly for camel/pascal
                resolved.append(f"{name}{suffix}")
            else:
                resolved.append(name)
                
        logger.info("Resolved schema collisions (duplicate names) successfully.")
        return resolved


    def optimize_metadata(self, raw_columns: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        THE MICRO-API (FOR NULLHUNTER ENGINE - PASS 1)
        
        Processes purely the metadata (List of Strings) to avoid O(N) DataFrame overhead.
        This is called EXACTLY ONCE on Chunk 1 to generate 'Asset 2'.
        
        Args:
            raw_columns (List[str]): The raw 2D string columns outputted by the Flattener.
            
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
            
        # 5. Ledger Generation
        translation_ledger = {new: old for new, old in zip(new_cols, raw_columns)}
        
        logger.info("Asset 2 Generated. Ready for O(1) deployment across all chunks.")
        return new_cols, translation_ledger


    def optimize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        THE MACRO-API (FOR STANDALONE SAAS USAGE)
        
        Allows independent Data Scientists to use this module on a standard Pandas DataFrame 
        without needing the entire NullHunter Engine.
        
        Args:
            df (pd.DataFrame): The target DataFrame.
            
        Returns:
            Tuple[pd.DataFrame, Dict]: The mutated DataFrame and its Translation Ledger.
        """
        logger.info("Standalone Mode: Optimizing DataFrame schema inplace...")
        
        # Extract raw columns, route through Brain, and apply
        raw_cols = list(df.columns)
        optimized_headers, ledger = self.optimize_metadata(raw_cols)
        
        # Apply instantly
        optimized_df = df.copy(deep=False)
        optimized_df.columns = optimized_headers
        
        return optimized_df, ledger


    