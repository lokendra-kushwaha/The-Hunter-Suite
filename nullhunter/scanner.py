import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union

# ==========================================
# LOGGER CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] NULLHUNTER_SCANNER: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataScanner:
    """
    The Ultimate SaaS-Level Data Profiler and Blueprint Generator (Layer 2).
    
    Acts as the 'Brain' of NullHunter. It is completely decoupled from the execution
    engine. It analyzes flat 2D DataFrames and generates a comprehensive JSON-like 
    Blueprint containing recommended cleaning strategies and memory optimizations.
    """

    def __init__(
        self,
        drop_threshold: float = 0.60,
        advanced_impute_threshold: float = 0.05,
        skewness_threshold: float = 1.0,
        cardinality_ratio: float = 0.40,
        outlier_z_threshold: float = 3.0,
        optimize_memory: bool = True,
        detect_text_anomalies: bool = True
    ) -> None:
        """
        Initializes the DataScanner with highly configurable statistical thresholds.
        
        Args:
            drop_threshold (float): If missing values exceed this (e.g., 60%), recommend dropping the column.
            advanced_impute_threshold (float): If missing > this (e.g., 5%), recommend KNN/MICE over simple mean.
            skewness_threshold (float): Absolute skewness limit to decide between mean (normal) and median (skewed).
            cardinality_ratio (float): (Unique values / Total values). Below this, treat text as categorical.
            outlier_z_threshold (float): Z-score absolute limit to flag a row as an outlier.
            optimize_memory (bool): Whether to generate downcasting/dtype memory optimization rules.
            detect_text_anomalies (bool): Whether to scan strings for whitespaces, special chars, etc.
        """
        self.drop_threshold = drop_threshold
        self.advanced_impute_threshold = advanced_impute_threshold
        self.skewness_threshold = skewness_threshold
        self.cardinality_ratio = cardinality_ratio
        self.outlier_z_threshold = outlier_z_threshold
        self.optimize_memory = optimize_memory
        self.detect_text_anomalies = detect_text_anomalies
        
        logger.debug("DataScanner initialized with advanced statistical parameters.")

    def _analyze_memory(self, series: pd.Series, series_type: str) -> Optional[str]:
        """
        Analyzes the column and recommends the most memory-efficient data type.
        
        Args:
            series (pd.Series): The data column.
            series_type (str): General type ('numeric' or 'object').
            
        Returns:
            Optional[str]: The recommended casting command (e.g., 'cast:int8', 'cast:category').
        """
        if not self.optimize_memory or series.empty:
            return None

        # 1. Memory Optimization for Numerics
        if series_type == 'numeric':
            col_min, col_max = series.min(), series.max()
            
            # Integer Downcasting
            if pd.api.types.is_integer_dtype(series):
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    return "cast:int8"
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    return "cast:int16"
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    return "cast:int32"
                    
            # Float Downcasting
            elif pd.api.types.is_float_dtype(series):
                if col_min >= np.finfo(np.float32).min and col_max <= np.finfo(np.float32).max:
                    return "cast:float32"
                    
        # 2. Memory Optimization for Objects/Strings
        elif series_type == 'object':
            num_unique = series.nunique()
            total_count = len(series.dropna())
            
            if total_count > 0:
                unique_ratio = num_unique / total_count
                # If unique values are few (e.g. 'Gender', 'Country'), use Category dtype
                if unique_ratio < self.cardinality_ratio:
                    return "cast:category"
                # If mostly unique (like Names, Sentences), use PyArrow String for speed
                else:
                    return "cast:string[pyarrow]"
                    
        return None

    def _analyze_numeric(self, series: pd.Series, missing_ratio: float) -> List[str]:
        """
        Analyzes a numeric column for outliers and missing value imputation strategies.
        
        Args:
            series (pd.Series): The numeric data column.
            missing_ratio (float): Percentage of missing values (0.0 to 1.0).
            
        Returns:
            List[str]: A list of recommended cleaning layers (e.g., ['outlier_capper', 'median_imputer']).
        """
        recommendations = []
        
        # 1. Outlier Detection (Z-Score approximation ignoring NaNs)
        if len(series.dropna()) > 3: # Need minimum data for variance
            std_dev = series.std()
            if std_dev > 0:
                z_scores = np.abs((series - series.mean()) / std_dev)
                if (z_scores > self.outlier_z_threshold).any():
                    recommendations.append("outlier_capper")

        # 2. Missing Value Imputation
        if missing_ratio > 0:
            skewness = series.skew()
            
            # High missing ratio: Use Advanced ML Imputers
            if missing_ratio > self.advanced_impute_threshold:
                if abs(skewness) > self.skewness_threshold:
                    recommendations.append("knn_imputer") # Better for skewed non-linear relationships
                else:
                    recommendations.append("mice_imputer") # Better for normally distributed linear data
                    
            # Low missing ratio: Use Simple Statistical Imputers
            else:
                if abs(skewness) > self.skewness_threshold:
                    recommendations.append("median_imputer")
                else:
                    recommendations.append("mean_imputer")
                    
        return recommendations

    def _analyze_categorical(self, series: pd.Series, missing_ratio: float) -> List[str]:
        """
        Analyzes text/categorical columns for anomalies and imputation.
        
        Args:
            series (pd.Series): The text data column.
            missing_ratio (float): Percentage of missing values (0.0 to 1.0).
            
        Returns:
            List[str]: Recommended cleaning layers (e.g., ['strip_whitespace', 'mode_imputer']).
        """
        recommendations = []
        
        # 1. Text Anomalies (Leading/Trailing spaces, etc.)
        if self.detect_text_anomalies and len(series.dropna()) > 0:
            sample = series.dropna().astype(str)
            # Check if any string starts or ends with whitespace
            if sample.str.match(r'^\s+|\s+$').any():
                recommendations.append("strip_whitespace")
                
            # Check for generic 'unknown' or '?' values acting as NaNs
            if sample.isin(['?', 'N/A', 'unknown', 'null']).any():
                recommendations.append("standardize_nulls")

        # 2. Missing Value Imputation
        if missing_ratio > 0:
            num_unique = series.nunique()
            total_count = len(series.dropna())
            unique_ratio = num_unique / total_count if total_count > 0 else 1.0
            
            # If it's a category (low cardinality), impute with Mode
            if unique_ratio < self.cardinality_ratio:
                recommendations.append("mode_imputer")
            # If it's highly unique (like IDs, Names), we can't impute it accurately
            else:
                recommendations.append("flag_missing_text")
                
        return recommendations

    def scan(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        The Master Profiling Engine.
        Iterates over the dataset and builds the final JSON Blueprint.
        
        Args:
            df (pd.DataFrame): The 2D Flat DataFrame to scan.
            
        Returns:
            Dict[str, Dict[str, Any]]: The Master Blueprint containing cleaning and memory rules.
        """
        logger.info(f"Initiating Smart Scan on DataFrame (Shape: {df.shape})...")
        blueprint: Dict[str, Dict[str, Any]] = {}
        total_rows = len(df)
        
        if total_rows == 0:
            logger.warning("DataScanner received an empty DataFrame.")
            return blueprint

        for col in df.columns:
            series = df[col]
            missing_count = series.isna().sum()
            missing_ratio = missing_count / total_rows
            
            # Initialize Blueprint node for this column
            blueprint[col] = {
                "cleaning_pipeline": [],
                "memory_optimization": None
            }
            
            # --- RULE 1: The Fatal Drop ---
            if missing_ratio >= self.drop_threshold:
                blueprint[col]["cleaning_pipeline"].append("drop_column")
                continue # Skip further analysis for this dropped column
                
            # --- RULE 2: Single Value Uselessness ---
            if series.nunique() <= 1 and missing_ratio == 0:
                blueprint[col]["cleaning_pipeline"].append("drop_constant_column")
                continue
                
            # --- RULE 3: Routing by Data Type ---
            is_numeric = pd.api.types.is_numeric_dtype(series)
            
            if is_numeric:
                pipeline = self._analyze_numeric(series, missing_ratio)
                blueprint[col]["cleaning_pipeline"].extend(pipeline)
                blueprint[col]["memory_optimization"] = self._analyze_memory(series, 'numeric')
                
            else:
                pipeline = self._analyze_categorical(series, missing_ratio)
                blueprint[col]["cleaning_pipeline"].extend(pipeline)
                blueprint[col]["memory_optimization"] = self._analyze_memory(series, 'object')

        logger.info("Data profiling complete. Blueprint generated successfully.")
        return blueprint


# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # Simulated User Experience:
    # 1. User loads data
    dummy_df = pd.DataFrame({'Age': [25, 800, np.nan, 30], 'Gender': [' Male ', 'Female', '?', 'Male']})
    
    # 2. User invokes Scanner standalone
    scanner = DataScanner()
    my_blueprint = scanner.scan(dummy_df)
    print(my_blueprint)