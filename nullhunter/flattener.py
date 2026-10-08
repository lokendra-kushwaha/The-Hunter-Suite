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
    Enterprise-Grade Structural Schema Encoder (Layer 1.5).

    Operates in a Dual-Pass architecture. In Pass 1 (Discovery), it acts as the 'Brain', 
    analyzing complex 3D hierarchical structures (MultiIndex) and generating a precise 
    Reverse Shape Map (Asset 1). In Pass 2 (Execution), it acts as a 'Dumb Applicator', 
    bypassing all computation to apply pre-optimized metadata in O(1) time complexity.

    Attributes:
        delimiter (str): Character used to join hierarchical column levels.
        col_strategy (str): 'join', 'top', or 'bottom' for multi-level resolution.
        reset_row_index (bool): Whether to demote hierarchical rows to columns.
        handle_sparse (bool): If True, replaces NaN/NaT in headers with 'unnamed'.
    """

    def __init__(
        self, 
        delimiter: str = "_", 
        col_strategy: str = "join",
        reset_row_index: bool = True,
        handle_sparse: bool = True
    ) -> None:
        """
        Initializes the SchemaFlattener with advanced structural parameters.

        Args:
            delimiter (str): The string to use to join MultiIndex levels.
            col_strategy (str): 'join' (all levels), 'top' (highest level), or 'bottom' (deepest).
            reset_row_index (bool): If True, transforms Row MultiIndex into standard columns.
            handle_sparse (bool): If True, cleans empty or sparse index levels dynamically.
        """
        self.delimiter = delimiter
        self.col_strategy = col_strategy.lower()
        self.reset_row_index = reset_row_index
        self.handle_sparse = handle_sparse
        
        if self.col_strategy not in ['join', 'top', 'bottom']:
            raise ValueError(
                f"[NullHunter Fatal] Invalid col_strategy '{self.col_strategy}'. "
                "Allowed: 'join', 'top', 'bottom'."
            )

        self.schema_blueprint: Dict[str, Any] = {}
        logger.info(f"SchemaFlattener Armed. Strategy: {self.col_strategy.upper()}")


    def _generate_flat_columns(self, multi_columns: pd.MultiIndex) -> Tuple[List[str], Dict[str, Tuple]]:
        """
        Calculates the 2D column projection and strictly maps the original 3D coordinates.

        Args:
            multi_columns (pd.MultiIndex): The complex 3D pandas index.

        Returns:
            Tuple[List[str], Dict[str, Tuple]]: 
                - List of raw 2D string columns.
                - Dict mapping the new 2D string to the original 3D tuple (Shape Map).
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
                        if self.handle_sparse:
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
        PASS 1 (DISCOVERY MODE): Analyzes the initial chunk to capture structural DNA.
        Computes the Shape Map (Asset 1) for the Reconstructor.

        Args:
            df (pd.DataFrame): The very first chunk from the DataLoader.

        Returns:
            Dict[str, Any]: The structural metadata blueprint including the Shape Map.
        """
        is_col_multi = isinstance(df.columns, pd.MultiIndex)
        is_row_multi = isinstance(df.index, pd.MultiIndex)
        
        schema = {
            'is_col_multiindex': is_col_multi,
            'is_row_multiindex': is_row_multi,
            'original_col_names': df.columns.names if is_col_multi else list(df.columns),
            'original_row_names': df.index.names,
            'needs_flattening': is_col_multi or (is_row_multi and self.reset_row_index),
            'raw_2d_columns': None,
            'reconstruction_mapping': None  # ASSET 1
        }
        
        if is_col_multi:
            flat_cols, mapping_dict = self._generate_flat_columns(df.columns)
            schema['raw_2d_columns'] = flat_cols
            schema['reconstruction_mapping'] = mapping_dict
            
        self.schema_blueprint = schema
        return schema


    def transform(
        self, 
        df: pd.DataFrame, 
        precomputed_headers: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        PASS 2 (BYPASS/EXECUTION MODE): O(1) latency structural projection.
        Applies pre-optimized metadata instantly without recalculating transformations.

        Args:
            df (pd.DataFrame): The raw chunk to flatten.
            precomputed_headers (Optional[List[str]]): Asset 2 from the Optimizer.

        Returns:
            pd.DataFrame: A strict 2D representation of the data.
        """
        if not self.schema_blueprint:
            raise RuntimeError("Schema blueprint missing. Execute generate_schema() in Pass 1 first.")
            
        if not self.schema_blueprint['needs_flattening'] and not precomputed_headers:
            return df

        # Shallow copy guarantees zero RAM duplication for the data body
        flat_df = df.copy(deep=False)
        
        # O(1) Row Demotion
        if self.schema_blueprint['is_row_multiindex'] and self.reset_row_index:
            flat_df = flat_df.reset_index(drop=False)

        # O(1) Metadata Assignment (Bypass Mode)
        if precomputed_headers:
            # Absolute bypass: apply Asset 2 directly from Engine
            flat_df.columns = precomputed_headers
        elif self.schema_blueprint['is_col_multiindex']:
            # Fallback for Pass 1 usage before Optimizer generates Asset 2
            flat_df.columns = self.schema_blueprint['raw_2d_columns']

        return flat_df
    