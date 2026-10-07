import re
import logging
import pandas as pd
import difflib
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
    
    This module executes strictly after the Schema Flattener (Layer 1.5) and before 
    the Core Engine/Scanner. It provides maximum user control over the dataset's 
    semantics (headers and indexes) without destroying the Chain of Custody.
    
    It returns the optimized DataFrame along with a 'Translation Ledger' (a reverse 
    mapping dictionary) so the Master Engine can synchronize state changes and 
    prevent the downstream Reconstructor from crashing.
    """

    def __init__(
        self,
        clean_names: bool = True,
        resolve_duplicates: bool = True,
        ai_auto_correct: bool = False,
        custom_columns: Optional[Union[List[str], Dict[str, str]]] = None,
        custom_index: Optional[Union[str, List[Any]]] = None,
        auto_correct_threshold: float = 0.75,
        domain_dictionary: Optional[List[str]] = None
    ) -> None:
        """
        Initializes the SchemaOptimizer with highly flexible SaaS-level parameters.
        
        Args:
            clean_names (bool): Strips whitespaces, special chars, and converts to snake_case.
            resolve_duplicates (bool): Appends numerical suffixes to prevent Pandas collision (e.g., sales_1, sales_2).
            ai_auto_correct (bool): Uses fuzzy string matching to fix common typos (e.g., 'gendre' -> 'gender').
            custom_columns (Union[List[str], Dict[str, str], None]): 
                - If List: Overwrites the entire column schema (must match exact column count).
                - If Dict: Maps specific old column names to new ones {old: new}.
            custom_index (Union[str, List[Any], None]):
                - If str: Extracts the specified column to become the new DataFrame index.
                - If List: Applies the external list as the new index.
            auto_correct_threshold (float): Similarity threshold (0.0 to 1.0) for AI fuzzy matching.
            domain_dictionary (Optional[List[str]]): A custom list of valid standard column names for AI matching.
        """
        self.clean_names = clean_names
        self.resolve_duplicates = resolve_duplicates
        self.ai_auto_correct = ai_auto_correct
        self.custom_columns = custom_columns
        self.custom_index = custom_index
        
        # AI/Fuzzy Matching Configuration
        self.auto_correct_threshold = auto_correct_threshold
        self.standard_dictionary = domain_dictionary or [
            'id', 'name', 'first_name', 'last_name', 'age', 'gender', 'sex', 
            'email', 'phone', 'address', 'city', 'state', 'zip', 'country', 
            'salary', 'income', 'revenue', 'profit', 'date', 'date_of_birth', 
            'dob', 'status', 'category', 'type', 'amount', 'quantity', 'price'
        ]
        
        logger.debug("SchemaOptimizer initialized with SaaS-level control parameters.")

    def _clean_column_string(self, col_name: str) -> str:
        """
        Converts a messy string into a pristine, SQL-ready snake_case format.
        """
        if pd.isna(col_name) or not str(col_name).strip():
            return "unnamed_col"
            
        # Convert to string, lower case, replace any non-alphanumeric with underscore
        cleaned = re.sub(r'[^a-zA-Z0-9]', '_', str(col_name).strip().lower())
        # Collapse multiple underscores into one and strip from ends
        cleaned = re.sub(r'_+', '_', cleaned).strip('_')
        
        return cleaned if cleaned else "unnamed_col"

    def _apply_ai_auto_correct(self, col_name: str) -> str:
        """
        Simulates an AI semantic fixer using difflib for fuzzy string matching.
        Fixes typos based on the standard domain dictionary.
        """
        # Get closest matches based on the threshold
        matches = difflib.get_close_matches(
            col_name, 
            self.standard_dictionary, 
            n=1, 
            cutoff=self.auto_correct_threshold
        )
        if matches:
            logger.info(f"[AI Semantic Fix] Auto-corrected '{col_name}' ➔ '{matches[0]}'")
            return matches[0]
        return col_name

    def _resolve_column_duplicates(self, cols: List[str]) -> List[str]:
        """
        Safely resolves naming collisions by appending iterative numeric suffixes.
        """
        counts = Counter(cols)
        if not any(count > 1 for count in counts.values()):
            return cols  # No duplicates

        seen = {}
        resolved = []
        for name in cols:
            if counts[name] > 1:
                seen[name] = seen.get(name, 0) + 1
                resolved_name = f"{name}_{seen[name]}"
                resolved.append(resolved_name)
            else:
                resolved.append(name)
                
        logger.info("Resolved duplicate column names to prevent system crashes.")
        return resolved

    def _handle_custom_columns(self, current_cols: List[str]) -> List[str]:
        """
        Applies user-defined lists or dictionaries to override the current headers.
        """
        if isinstance(self.custom_columns, list):
            if len(self.custom_columns) != len(current_cols):
                raise NullHunterSchemaError(
                    f"Length mismatch: custom_columns list has {len(self.custom_columns)} items, "
                    f"but dataset has {len(current_cols)} columns."
                )
            logger.info("Applied full custom column list override.")
            return self.custom_columns
            
        elif isinstance(self.custom_columns, dict):
            # Map specific old names to new names; keep old if not in dict
            mapped_cols = [self.custom_columns.get(c, c) for c in current_cols]
            logger.info("Applied custom column dictionary mapping.")
            return mapped_cols
            
        return current_cols

    def optimize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        The Master Execution Function.
        Applies all transformations dynamically and generates the Translation Ledger.
        
        Args:
            df (pd.DataFrame): The flattened 2D DataFrame chunk from Layer 1.5.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, str]]: 
                - The semantically optimized DataFrame.
                - The Translation Ledger {new_col_name: original_flat_name}.
        """
        logger.info("Initiating Semantic Schema Optimization...")
        
        optimized_df = df.copy(deep=False)
        original_cols = list(optimized_df.columns)
        new_cols = original_cols.copy()
        
        # 1. Apply Custom Columns Override
        if self.custom_columns is not None:
            new_cols = self._handle_custom_columns(new_cols)
            
        # 2. Apply Standard Cleaning (SQL Readiness)
        if self.clean_names:
            new_cols = [self._clean_column_string(c) for c in new_cols]
            
        # 3. Apply AI Semantic Typo Fixer
        if self.ai_auto_correct:
            new_cols = [self._apply_ai_auto_correct(c) for c in new_cols]
            
        # 4. Resolve Duplicates (Crucial for Pandas Stability)
        if self.resolve_duplicates:
            new_cols = self._resolve_column_duplicates(new_cols)
            
        # 5. Apply the optimized columns to the DataFrame
        optimized_df.columns = new_cols
        
        # 6. Apply Custom Index Logic
        if self.custom_index is not None:
            if isinstance(self.custom_index, str):
                if self.custom_index in optimized_df.columns:
                    optimized_df = optimized_df.set_index(self.custom_index, drop=True)
                    logger.info(f"Row index set dynamically to column '{self.custom_index}'.")
                else:
                    raise NullHunterSchemaError(f"Custom index column '{self.custom_index}' not found in dataset.")
            elif isinstance(self.custom_index, list):
                if len(self.custom_index) == len(optimized_df):
                    optimized_df.index = self.custom_index
                    logger.info("External list applied as custom row index.")
                else:
                    raise NullHunterSchemaError("Length mismatch between custom_index list and DataFrame rows.")

        # 7. Generate the Master Translation Ledger
        # Format: { 'new_optimized_name' : 'original_flat_name' }
        # The Engine will merge this with the Flattener's blueprint.
        translation_ledger = {new: old for new, old in zip(new_cols, original_cols)}
        
        logger.info("Schema Optimization complete. Translation Ledger generated successfully.")
        return optimized_df, translation_ledger