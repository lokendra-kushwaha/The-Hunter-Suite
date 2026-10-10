"""
NullHunter Schema Flattener (The Structural Encoder)
================================================================

Operates via a Dual-Pass architecture to achieve Zero-Cost Abstraction.
- Pass 1 (Discovery): Analyzes complex 3D hierarchical structures (MultiIndex) 
  to generate a precise Reverse Shape Map (Asset 1), securing both column and row ledgers.
- Pass 2 (Execution): Acts as a 'Dumb Applicator', bypassing heavy computations 
  to apply pre-optimized metadata in O(1) time complexity per chunk.

Key Architectural Upgrades:
1. Reconstructor Synchronization: Strictly enforces the `__index_names__` protocol.
2. Collision Resolution: Auto-resolves naming conflicts during multidimensional flattening.
3. Standalone Capability: Can be utilized entirely outside the framework via `flatten()`.
"""

import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Set

# ==========================================
# LOGGER CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] NULLHUNTER_FLATTENER: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class NullHunterCollisionError(Exception):
    """Raised when flattening results in duplicate column names and strict mode is active."""
    pass


class SchemaFlattener:
    """
    Enterprise-Grade Structural Schema Encoder.

    Attributes:
        delimiter (str): Sequence used to join hierarchical column levels.
        col_strategy (str): Resolution strategy ('join', 'top', 'bottom').
        reset_row_index (bool): If True, demotes hierarchical rows to standard columns.
        handle_sparse (bool): If True, dynamically fills NaN/NaT tuple levels.
        drop_empty_levels (bool): If True, ignores empty levels instead of naming them.
        memory_safe_copy (bool): If True, enforces shallow copying to prevent RAM duplication.
        collision_strategy (str): How to handle duplicate flat names ('rename' or 'raise').
        strip_whitespace (bool): Cleans messy strings in tuple levels before joining.
        unnamed_prefix (str): The specific flag used for blank levels (matches Reconstructor).
    """

    def __init__(
        self, 
        delimiter: str = "_", 
        col_strategy: str = "join",
        reset_row_index: bool = True,
        handle_sparse: bool = True,
        drop_empty_levels: bool = False,
        memory_safe_copy: bool = True,
        collision_strategy: str = "rename",
        strip_whitespace: bool = True,
        unnamed_prefix: str = "UNNAMED_LEVEL"
    ) -> None:
        """
        Initializes the SchemaFlattener with advanced, SaaS-ready structural parameters.

        Args:
            delimiter (str): The string to use to join MultiIndex levels. Defaults to '_'.
            col_strategy (str): 'join' (all levels), 'top' (highest), or 'bottom' (deepest).
            reset_row_index (bool): Whether to transform Row MultiIndex into standard columns.
            handle_sparse (bool): Whether to replace blank/NaN index levels dynamically.
            drop_empty_levels (bool): Whether to drop blank levels entirely.
            memory_safe_copy (bool): Whether to use shallow copies during Pass 2.
            collision_strategy (str): Action on name conflicts ('rename', 'raise').
            strip_whitespace (bool): Trims spaces from level names before concatenation.
            unnamed_prefix (str): Placeholder for empty levels. Defaults to 'UNNAMED_LEVEL'.
        """
        self.delimiter = delimiter
        self.col_strategy = col_strategy.lower()
        self.reset_row_index = reset_row_index
        self.handle_sparse = handle_sparse
        self.drop_empty_levels = drop_empty_levels
        self.memory_safe_copy = memory_safe_copy
        self.collision_strategy = collision_strategy.lower()
        self.strip_whitespace = strip_whitespace
        self.unnamed_prefix = unnamed_prefix
        
        if self.col_strategy not in ['join', 'top', 'bottom']:
            raise ValueError(f"Invalid col_strategy '{self.col_strategy}'. Allowed: 'join', 'top', 'bottom'.")
            
        if self.collision_strategy not in ['rename', 'raise']:
            raise ValueError(f"Invalid collision_strategy '{self.collision_strategy}'. Allowed: 'rename', 'raise'.")

        self.schema_blueprint: Dict[str, Any] = {}
        logger.debug(
            f"SchemaFlattener Armed. Strategy: {self.col_strategy.upper()} | "
            f"Memory Safe: {self.memory_safe_copy} | Collision Logic: {self.collision_strategy}"
        )

    def _generate_flat_columns(self, multi_columns: pd.MultiIndex) -> Tuple[List[str], Dict[str, Tuple]]:
        """
        Calculates the 2D column projection with strict Collision Detection.

        Args:
            multi_columns (pd.MultiIndex): The complex 3D pandas index.

        Returns:
            Tuple[List[str], Dict[str, Tuple]]: 
                - List of raw 2D string columns.
                - Dict mapping the new 2D string to the original 3D tuple (Column Ledger).
                
        Raises:
            NullHunterCollisionError: If duplicate columns are generated under 'raise' strategy.
        """
        flat_cols = []
        reverse_mapping = {}
        seen_names: Set[str] = set()
        
        for col_tuple in multi_columns.tolist():
            # Apply strategy extraction
            if self.col_strategy == 'top':
                target_levels = [col_tuple[0]]
            elif self.col_strategy == 'bottom':
                target_levels = [col_tuple[-1]]
            else:
                target_levels = list(col_tuple)
                
            valid_parts = []
            for part in target_levels:
                part_str = str(part)
                if self.strip_whitespace and part is not None:
                    part_str = part_str.strip()
                    
                if pd.isna(part) or not part_str:
                    if self.drop_empty_levels:
                        continue
                    elif self.handle_sparse:
                        valid_parts.append(self.unnamed_prefix)
                else:
                    valid_parts.append(part_str)
                    
            raw_name = self.delimiter.join(valid_parts)
            flat_name = raw_name if raw_name else self.unnamed_prefix
            
            # --- Enterprise Collision Detection Engine ---
            original_flat_name = flat_name
            collision_counter = 1
            
            while flat_name in seen_names:
                if self.collision_strategy == 'raise':
                    raise NullHunterCollisionError(
                        f"Flattening caused a naming collision on '{flat_name}'. "
                        "Switch collision_strategy to 'rename' to auto-resolve."
                    )
                # Auto-resolve by appending suffix (e.g., User_ID_1)
                flat_name = f"{original_flat_name}{self.delimiter}{collision_counter}"
                collision_counter += 1
                
            seen_names.add(flat_name)
            flat_cols.append(flat_name)
            reverse_mapping[flat_name] = col_tuple
            
        return flat_cols, reverse_mapping

    def generate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PASS 1 (DISCOVERY MODE): Computes Asset 1 (The Master Shape Map).
        Now strictly enforces the `__index_names__` protocol for the Reconstructor.

        Args:
            df (pd.DataFrame): The definitive first chunk (Probe) from the DataLoader.

        Returns:
            Dict[str, Any]: The structural metadata blueprint including Asset 1.
        """
        logger.info("Executing Schema Discovery (Pass 1). Constructing Master Shape Map...")
        
        is_col_multi = isinstance(df.columns, pd.MultiIndex)
        is_row_multi = isinstance(df.index, pd.MultiIndex)
        
        original_col_names = list(df.columns.names) if is_col_multi else list(df.columns)
        original_row_names = list(df.index.names)
        
        schema = {
            'is_col_multiindex': is_col_multi,
            'is_row_multiindex': is_row_multi,
            'original_col_names': original_col_names,
            'original_row_names': original_row_names,
            'needs_flattening': is_col_multi or (is_row_multi and self.reset_row_index),
            'raw_2d_columns': None,
            'reconstruction_mapping': {}  # Unified Ledger
        }
        
        # 1. Process Column Ledger
        if is_col_multi:
            flat_cols, mapping_dict = self._generate_flat_columns(df.columns)
            schema['raw_2d_columns'] = flat_cols
            schema['reconstruction_mapping'].update(mapping_dict)
            
        # 2. Process Row Ledger (CRITICAL FIX FOR RECONSTRUCTOR)
        index_registry = []
        if is_row_multi and self.reset_row_index:
            for i, name in enumerate(original_row_names):
                safe_name = str(name).strip() if name is not None else f"{self.unnamed_prefix}_ROW_{i}"
                index_registry.append(safe_name)
                # Map the generated 2D column name back to its original index tuple/string
                schema['reconstruction_mapping'][safe_name] = name if name is not None else safe_name
                
        # Inject the secret Reconstructor Protocol Key
        schema['reconstruction_mapping']['__index_names__'] = index_registry

        self.schema_blueprint = schema
        logger.info("Asset 1 (Master Shape Map & Ledgers) successfully secured.")
        return schema

    def transform(
        self, 
        df: pd.DataFrame, 
        precomputed_headers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        PASS 2 (EXECUTION MODE): O(1) projection onto a data chunk.

        Args:
            df (pd.DataFrame): The raw DataFrame chunk.
            precomputed_headers (Optional[List[str]]): Asset 2 generated by the SchemaOptimizer. 
                If provided, absolutely bypasses computation. Defaults to None.

        Returns:
            pd.DataFrame: A strict 2D representation of the data chunk.

        Raises:
            RuntimeError: If called before `generate_schema()`.
        """
        if not self.schema_blueprint:
            raise RuntimeError("[NullHunter Fatal] Blueprint missing. Execute generate_schema() first.")
            
        if not self.schema_blueprint['needs_flattening'] and not precomputed_headers:
            return df

        flat_df = df.copy(deep=not self.memory_safe_copy)
        
        # 1. Row Ledger Execution: O(1) Demotion
        if self.schema_blueprint['is_row_multiindex'] and self.reset_row_index:
            flat_df = flat_df.reset_index(drop=False)

        # 2. Column Ledger Execution: O(1) Assignment
        if precomputed_headers:
            flat_df.columns = precomputed_headers
        elif self.schema_blueprint['is_col_multiindex']:
            final_cols = self.schema_blueprint['raw_2d_columns'].copy()
            
            if self.schema_blueprint['is_row_multiindex'] and self.reset_row_index:
                # Retrieve the exact names we logged in Pass 1 for perfect length matching
                row_names = self.schema_blueprint['reconstruction_mapping']['__index_names__']
                final_cols = row_names + final_cols
                
            flat_df.columns = final_cols

        return flat_df

    def flatten(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        The Standalone Public API. 
        Executes Discovery (Pass 1) and Transformation (Pass 2) sequentially in one call.
        Perfect for utilizing the module outside the NullHunter pipeline.

        Args:
            df (pd.DataFrame): The target DataFrame.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 
                - The 2D flattened DataFrame.
                - The Master Ledger (`reconstruction_mapping`) required by the Reconstructor.
        """
        logger.info("Standalone Mode Activated: Executing one-shot flatten operation.")
        schema = self.generate_schema(df)
        flat_df = self.transform(df)
        return flat_df, schema.get('reconstruction_mapping', {})