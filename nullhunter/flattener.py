import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List, Union

# ==========================================
# LOGGER CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] NULLHUNTER_FLATTENER: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SchemaFlattener:
    """
    Structural Schema Encoder (Layer 1.5).

    Operates via a Dual-Pass architecture to achieve Zero-Cost Abstraction.
    Pass 1 (Discovery): Analyzes complex 3D hierarchical structures (MultiIndex for both 
    rows and columns) to generate a precise Reverse Shape Map (Asset 1), capturing both 
    column mappings and row ledgers.
    Pass 2 (Execution): Acts as a 'Dumb Applicator', bypassing heavy tuple-merging 
    computations to apply pre-optimized metadata in O(1) time complexity.

    Attributes:
        delimiter (str): Character sequence used to join hierarchical column levels.
        col_strategy (str): Strategy for multi-level resolution ('join', 'top', 'bottom').
        reset_row_index (bool): If True, safely demotes hierarchical rows to standard columns.
        handle_sparse (bool): If True, dynamically fills NaN/NaT empty tuple levels.
        drop_empty_levels (bool): If True, completely ignores empty levels instead of naming them.
        memory_safe_copy (bool): If True, enforces shallow copying to prevent RAM duplication.
    """

    def __init__(
        self, 
        delimiter: str = "_", 
        col_strategy: str = "join",
        reset_row_index: bool = True,
        handle_sparse: bool = True,
        drop_empty_levels: bool = False,
        memory_safe_copy: bool = True
    ) -> None:
        """
        Initializes the SchemaFlattener with advanced, SaaS-ready structural parameters.

        Args:
            delimiter (str): The string to use to join MultiIndex levels (default: '_').
            col_strategy (str): 'join' (all valid levels), 'top' (highest), or 'bottom' (deepest).
            reset_row_index (bool): Whether to transform Row MultiIndex into standard columns.
            handle_sparse (bool): Whether to replace blank/NaN index levels dynamically.
            drop_empty_levels (bool): Whether to drop blank levels entirely instead of filling them.
            memory_safe_copy (bool): Whether to use shallow copies during Pass 2 to save RAM.
        """
        self.delimiter = delimiter
        self.col_strategy = col_strategy.lower()
        self.reset_row_index = reset_row_index
        self.handle_sparse = handle_sparse
        self.drop_empty_levels = drop_empty_levels
        self.memory_safe_copy = memory_safe_copy
        
        if self.col_strategy not in ['join', 'top', 'bottom']:
            raise ValueError(
                f"[NullHunter Fatal] Invalid col_strategy '{self.col_strategy}'. "
                "Allowed options: 'join', 'top', 'bottom'."
            )

        self.schema_blueprint: Dict[str, Any] = {}
        logger.info(
            f"SchemaFlattener Armed. Strategy: {self.col_strategy.upper()} | "
            f"Memory Safe: {self.memory_safe_copy}"
        )


    def _generate_flat_columns(self, multi_columns: pd.MultiIndex) -> Tuple[List[str], Dict[str, Tuple]]:
        """
        Calculates the 2D column projection and strictly maps the original 3D coordinates.

        Args:
            multi_columns (pd.MultiIndex): The complex 3D pandas index.

        Returns:
            Tuple[List[str], Dict[str, Tuple]]: 
                - List of raw 2D string columns.
                - Dict mapping the new 2D string to the original 3D tuple (Column Ledger).
        """
        flat_cols = []
        reverse_mapping = {}
        
        for col_tuple in multi_columns.tolist():
            if self.col_strategy == 'top':
                raw_name = col_tuple[0]
            elif self.col_strategy == 'bottom':
                raw_name = col_tuple[-1]
            else:
                valid_parts = []
                for part in col_tuple:
                    if pd.isna(part) or str(part).strip() == '':
                        if self.drop_empty_levels:
                            continue
                        elif self.handle_sparse:
                            valid_parts.append("unnamed")
                    else:
                        valid_parts.append(str(part))
                raw_name = self.delimiter.join(valid_parts)
                
            flat_name = str(raw_name).strip()
            if not flat_name:
                flat_name = "unnamed_level"
                
            flat_cols.append(flat_name)
            reverse_mapping[flat_name] = col_tuple
            
        return flat_cols, reverse_mapping


    def generate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        PASS 1 (DISCOVERY MODE): Analyzes the initial chunk to capture full structural DNA.
        Computes Asset 1 (The Master Shape Map), securing both Column and Row Ledgers.

        Args:
            df (pd.DataFrame): The definitive first chunk from the DataLoader.

        Returns:
            Dict[str, Any]: The structural metadata blueprint including Asset 1.
        """
        logger.info("Executing Schema Discovery (Pass 1). Constructing Master Shape Map...")
        
        is_col_multi = isinstance(df.columns, pd.MultiIndex)
        is_row_multi = isinstance(df.index, pd.MultiIndex)
        
        # Isolate original names for the Row/Column Ledgers
        original_col_names = list(df.columns.names) if is_col_multi else list(df.columns)
        original_row_names = list(df.index.names)
        
        schema = {
            'is_col_multiindex': is_col_multi,
            'is_row_multiindex': is_row_multi,
            'original_col_names': original_col_names,
            'original_row_names': original_row_names,  # The Row Ledger
            'needs_flattening': is_col_multi or (is_row_multi and self.reset_row_index),
            'raw_2d_columns': None,
            'reconstruction_mapping': None  # The Column Ledger
        }
        
        if is_col_multi:
            flat_cols, mapping_dict = self._generate_flat_columns(df.columns)
            schema['raw_2d_columns'] = flat_cols
            schema['reconstruction_mapping'] = mapping_dict
            
        self.schema_blueprint = schema
        logger.info("Asset 1 (Master Shape Map & Ledgers) successfully captured.")
        return schema


    def transform(
            self, 
            df: pd.DataFrame, 
            precomputed_headers: Optional[List[str]] = None
        ) -> pd.DataFrame:
            """
            Executes the structural projection onto a data chunk using the schema blueprint.
            
            Engineered for O(1) time complexity per chunk during the Execution Cycle (Pass 2).
            It seamlessly handles both 'Discovery Mode' fallback concatenation and 'Bypass Mode' 
            metadata assignment, ensuring zero length-mismatch collisions when rows are demoted.

            Args:
                df (pd.DataFrame): The raw DataFrame chunk extracted by the DataLoader.
                precomputed_headers (Optional[List[str]]): Asset 2 generated by the SchemaOptimizer. 
                    If provided (Pass 2), the function absolute-bypasses all internal logic and 
                    assigns these headers in O(1) time. Defaults to None (Pass 1).

            Returns:
                pd.DataFrame: A strict, perfectly aligned 2D representation of the data chunk 
                ready for downstream execution or optimization.

            Raises:
                RuntimeError: If called before `generate_schema()` has populated the blueprint.
                ValueError: If the length of precomputed_headers does not match the chunk's 
                    column count after row demotion (Pandas native error passed through).

            Notes:
                - During Pass 1 (precomputed_headers=None), if hierarchical rows were demoted 
                via `reset_index`, this method dynamically calculates unnamed level names 
                (e.g., 'level_0') and prepends them to the flattened 2D columns list to 
                prevent Pandas `ValueError: Length mismatch` exceptions.
                - Utilizes shallow copying (`deep=False`) based on `memory_safe_copy` state 
                to prevent RAM duplication for massive chunks.
            """
            if not self.schema_blueprint:
                raise RuntimeError(
                    "[NullHunter Fatal] Schema blueprint missing. Execute generate_schema() first."
                )
                
            if not self.schema_blueprint['needs_flattening'] and not precomputed_headers:
                return df

            # Execute Memory-Safe Copying to prevent RAM ballooning on large chunks
            flat_df = df.copy(deep=not self.memory_safe_copy)
            
            # 1. Row Ledger Execution: O(1) Row Demotion
            if self.schema_blueprint['is_row_multiindex'] and self.reset_row_index:
                flat_df = flat_df.reset_index(drop=False)

            # 2. Column Ledger Execution: O(1) Metadata Assignment
            if precomputed_headers:
                # PASS 2: Absolute bypass. Inject Asset 2 directly from the Core Engine.
                flat_df.columns = precomputed_headers
            elif self.schema_blueprint['is_col_multiindex']:
                # PASS 1: Fallback generation before Optimizer creates Asset 2.
                # Must combine Demoted Row Names + Flattened Column Names to match new length.
                final_cols = self.schema_blueprint['raw_2d_columns']
                
                if self.schema_blueprint['is_row_multiindex'] and self.reset_row_index:
                    # Handle cases where the original row index lacked a specific name
                    row_names = [
                        str(name) if name is not None else f"level_{i}" 
                        for i, name in enumerate(self.schema_blueprint['original_row_names'])
                    ]
                    final_cols = row_names + final_cols
                    
                flat_df.columns = final_cols

            return flat_df

    