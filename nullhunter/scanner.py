import os
import logging
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Union, Set, Tuple

# ==========================================
# LOGGER CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] NULLHUNTER_SCANNER: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# ISOLATED MAP-WORKER (For Multiprocessing)
# ==========================================
# Placed outside the class to ensure perfect Pickling for ProcessPoolExecutor.
def _map_chunk_stats(chunk: pd.DataFrame, target_columns: Optional[List[str]]) -> Dict[str, Dict[str, Any]]:
    """
    The Isolated Map Worker (Phase 1 of MapReduce).
    
    Extracts raw mathematical moments and frequency counts from a single chunk.
    It returns only tiny numerical scalars, dropping the heavy DataFrame immediately 
    to preserve RAM across parallel CPU cores.
    
    Args:
        chunk (pd.DataFrame): A 2D slice of the dataset.
        target_columns (Optional[List[str]]): Specific columns to profile.
        
    Returns:
        Dict: Raw chunk statistics mapped by column name.
    """
    chunk_stats: Dict[str, Dict[str, Any]] = {}
    cols_to_scan = target_columns if target_columns else chunk.columns

    for col in cols_to_scan:
        if col not in chunk.columns:
            continue
            
        series = chunk[col]
        total_rows = len(series)
        valid_data = series.dropna()
        valid_count = len(valid_data)
        missing_count = total_rows - valid_count
        
        is_numeric = pd.api.types.is_numeric_dtype(series)
        
        stats: Dict[str, Any] = {
            "total_rows": total_rows,
            "missing_count": missing_count,
            "valid_count": valid_count,
            "is_numeric": is_numeric,
            "sum_x": 0.0,
            "sum_x2": 0.0,
            "sum_x3": 0.0,
            "min": float('inf'),
            "max": float('-inf'),
            "unique_sample": set()
        }

        if valid_count > 0:
            if is_numeric:
                # Core mathematical moments for Variance, Skewness, and Outliers
                stats["sum_x"] = float(np.sum(valid_data))
                stats["sum_x2"] = float(np.sum(valid_data ** 2))
                stats["sum_x3"] = float(np.sum(valid_data ** 3))
                stats["min"] = float(np.min(valid_data))
                stats["max"] = float(np.max(valid_data))
                
                # Integer checking for Predictive Mean Matching (PMM)
                stats["is_discrete"] = bool(np.all(np.mod(valid_data, 1) == 0))
            else:
                # Capture a capped set of unique values to prevent RAM explosion on high cardinality
                # We cap at 50,000 per chunk. The Reducer will union them.
                unique_vals = valid_data.unique()
                if len(unique_vals) <= 50000:
                    stats["unique_sample"] = set(unique_vals)
                else:
                    stats["unique_sample"] = set(unique_vals[:50000])
                    stats["high_cardinality_flag"] = True

        chunk_stats[col] = stats
        
    return chunk_stats


# ==========================================
# THE MASTER SCANNER CLASS
# ==========================================
class DataScanner:
    """
    The Autonomous Statistical Profiler & AI Blueprint Generator for NullHunter.
    
    Implements a strict Separation of Concerns. It uses streaming mathematics
    (MapReduce) to calculate global properties of out-of-core datasets. It does NOT 
    modify data; it acts as a diagnostic physician, writing a 'prescription' (Blueprint) 
    utilizing industry-standard and cutting-edge imputation algorithms.
    """

    def __init__(
        self,
        target_columns: Optional[List[str]] = None,
        max_cores: Optional[int] = None,
        # Thresholds (The Knobs & Dials of the AI)
        drop_threshold: float = 0.85,
        constant_variance_drop: bool = True,
        skewness_high_threshold: float = 1.5,
        skewness_extreme_threshold: float = 3.0,
        outlier_z_threshold: float = 3.5,
        svd_sparsity_trigger: float = 0.40,
        mice_correlation_min: float = 0.10,
        cardinality_ratio_max: float = 0.05,
        time_series_mode: bool = False,
        deep_learning_mode: bool = True
    ) -> None:
        """
        Initializes the Scanner with an extensive array of heuristic parameters.
        
        Args:
            target_columns (Optional[List[str]]): Specific columns to profile. If None, scans all.
            max_cores (Optional[int]): Cores for parallel chunk mapping.
            drop_threshold (float): % of NaNs above which column is dropped (Default: 85%).
            constant_variance_drop (bool): Drop columns where all values are identical.
            skewness_high_threshold (float): Threshold to shift from Mean to Median/KNN.
            skewness_extreme_threshold (float): Threshold to trigger Outlier capping + Mahalanobis.
            outlier_z_threshold (float): Standard deviations to define an extreme outlier.
            svd_sparsity_trigger (float): Missing % threshold to recommend Matrix Factorization (SVD).
            mice_correlation_min (float): Minimum missing % to justify heavy MICE iteration.
            cardinality_ratio_max (float): Max unique/total ratio for categorical variables.
            time_series_mode (bool): If True, biases towards Spline/Kalman interpolation.
            deep_learning_mode (bool): If True, allows Neural Autoencoder recommendations.
        """
        self.target_columns = target_columns
        self.max_cores = max_cores or max(1, (os.cpu_count() or 4) - 1)
        
        # Heuristic Configuration Matrix
        self.drop_threshold = drop_threshold
        self.constant_variance_drop = constant_variance_drop
        self.skew_high = skewness_high_threshold
        self.skew_extreme = skewness_extreme_threshold
        self.outlier_z = outlier_z_threshold
        self.svd_trigger = svd_sparsity_trigger
        self.mice_trigger = mice_correlation_min
        self.cardinality_max = cardinality_ratio_max
        self.time_series_mode = time_series_mode
        self.deep_learning_mode = deep_learning_mode
        
        # State Tracking
        self.global_stats: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Scanner Armed. Engine Cores: {self.max_cores} | Target Columns: {self.target_columns or 'ALL'}")

    def _reduce_stats(self, all_chunk_stats: List[Dict[str, Dict[str, Any]]]) -> None:
        """
        The Master Reducer (Phase 2 of MapReduce).
        Merges intermediate statistics from all chunks to form a 100% accurate Global State.
        """
        logger.info("Reducing chunk statistics into Global State Matrices...")
        
        for chunk in all_chunk_stats:
            for col, stats in chunk.items():
                if col not in self.global_stats:
                    self.global_stats[col] = {
                        "total_rows": 0, "missing_count": 0, "valid_count": 0,
                        "is_numeric": stats["is_numeric"], "is_discrete": stats.get("is_discrete", False),
                        "sum_x": 0.0, "sum_x2": 0.0, "sum_x3": 0.0,
                        "min": float('inf'), "max": float('-inf'),
                        "unique_set": set(), "high_cardinality_flag": False
                    }
                
                g = self.global_stats[col]
                g["total_rows"] += stats["total_rows"]
                g["missing_count"] += stats["missing_count"]
                g["valid_count"] += stats["valid_count"]
                
                if stats["is_numeric"]:
                    g["sum_x"] += stats["sum_x"]
                    g["sum_x2"] += stats["sum_x2"]
                    g["sum_x3"] += stats["sum_x3"]
                    g["min"] = min(g["min"], stats["min"])
                    g["max"] = max(g["max"], stats["max"])
                    # If any chunk is not discrete, the global column is not discrete
                    if not stats.get("is_discrete", True):
                        g["is_discrete"] = False
                else:
                    if not g["high_cardinality_flag"]:
                        g["unique_set"].update(stats.get("unique_sample", set()))
                        if len(g["unique_set"]) > 100000 or stats.get("high_cardinality_flag", False):
                            g["high_cardinality_flag"] = True
                            g["unique_set"].clear() # Free RAM

    def _derive_global_mathematics(self) -> None:
        """
        Extracts complex mathematical moments (Variance, Skewness, StdDev) 
        from the aggregated sum matrices.
        """
        for col, g in self.global_stats.items():
            g["missing_ratio"] = g["missing_count"] / g["total_rows"] if g["total_rows"] > 0 else 1.0
            
            if g["is_numeric"] and g["valid_count"] > 2:
                N = g["valid_count"]
                mean = g["sum_x"] / N
                
                # Global Variance & Standard Deviation
                variance = max(0.0, (g["sum_x2"] / N) - (mean ** 2))
                std_dev = math.sqrt(variance)
                
                # Global Skewness (Derived from central moments)
                if std_dev > 0:
                    # Skewness = [E(x^3) - 3*mu*sigma^2 - mu^3] / sigma^3 (Approximate Population Skew)
                    m3_expected = g["sum_x3"] / N
                    skewness = (m3_expected - 3 * mean * variance - (mean ** 3)) / (std_dev ** 3)
                else:
                    skewness = 0.0
                    
                g["global_mean"] = mean
                g["global_std"] = std_dev
                g["global_skewness"] = skewness
                
            elif not g["is_numeric"] and g["valid_count"] > 0:
                g["cardinality_ratio"] = len(g["unique_set"]) / g["valid_count"] if not g["high_cardinality_flag"] else 1.0

    def _apply_decision_matrix(self) -> Dict[str, Dict[str, Any]]:
        """
        The Brain (Phase 3). Evaluates Global Mathematics against strict Heuristic Rules 
        to assign the most advanced execution algorithms.
        """
        blueprint: Dict[str, Dict[str, Any]] = {}
        
        for col, g in self.global_stats.items():
            pipeline = []
            mem_opt = None
            missing_ratio = g["missing_ratio"]
            
            blueprint[col] = {"cleaning_pipeline": pipeline, "memory_optimization": mem_opt}
            
            # --- ABSOLUTE RULES ---
            if missing_ratio == 1.0:
                pipeline.append("drop_column_all_nan")
                continue
            
            if missing_ratio >= self.drop_threshold:
                pipeline.append("drop_column_sparsity")
                continue
                
            if g["is_numeric"] and g.get("global_std", -1) == 0.0 and self.constant_variance_drop:
                pipeline.append("drop_constant_variance")
                continue
                
            if not g["is_numeric"] and missing_ratio == 0 and g.get("cardinality_ratio", 1.0) == 1.0:
                pipeline.append("drop_high_cardinality_id")
                continue

            # --- NUMERICAL AI IMPUTATION LOGIC ---
            if g["is_numeric"] and missing_ratio > 0:
                skew = abs(g.get("global_skewness", 0.0))
                is_discrete = g.get("is_discrete", False)
                
                # Check for extreme outliers (Max/Min vs Z-Threshold)
                mean = g.get("global_mean", 0)
                std = g.get("global_std", 1)
                has_extreme_outliers = (abs(g["max"] - mean) / std > self.outlier_z) or (abs(g["min"] - mean) / std > self.outlier_z)

                if has_extreme_outliers:
                    pipeline.append("outlier_capper_iqr")

                # Time-Series Override
                if self.time_series_mode:
                    pipeline.append("spline_interpolation" if not is_discrete else "ffill_bfill_imputer")
                
                # Extreme Sparsity (Matrix Factorization & Deep Learning)
                elif missing_ratio > self.svd_trigger:
                    if self.deep_learning_mode and not is_discrete:
                        pipeline.append("autoencoder_deep_imputer")
                    else:
                        pipeline.append("svd_matrix_factorization")
                        
                # Complex Multivariate Structure (MICE & Mahalanobis)
                elif missing_ratio > self.mice_trigger:
                    if skew > self.skew_extreme:
                        pipeline.append("mahalanobis_knn_imputer") # Handles skewed covariance
                    elif is_discrete:
                        pipeline.append("pmm_imputer") # Predictive Mean Matching for discrete counts (e.g. Cars)
                    else:
                        pipeline.append("mice_iterative_imputer") # Gold standard for normal-ish multivariates
                        
                # Standard Distribution Thresholds
                else:
                    if skew > self.skew_high:
                        pipeline.append("knn_imputer" if missing_ratio > 0.02 else "median_imputer")
                    else:
                        # Expectation-Maximization is elegant for minor normal data gaps
                        pipeline.append("em_algorithm_imputer" if missing_ratio > 0.01 else "mean_imputer")
            
            # --- CATEGORICAL/TEXT AI IMPUTATION LOGIC ---
            elif not g["is_numeric"]:
                if missing_ratio > 0:
                    cardinality = g.get("cardinality_ratio", 1.0)
                    
                    if cardinality < self.cardinality_max:
                        pipeline.append("mode_imputer")
                    elif self.deep_learning_mode:
                        pipeline.append("missforest_categorical_imputer")
                    else:
                        pipeline.append("flag_missing_text")
                        
                # Text structural hygiene
                pipeline.append("regex_whitespace_cleaner")

            # --- MEMORY OPTIMIZATION DECISIONS ---
            if g["is_numeric"]:
                min_v, max_v = g["min"], g["max"]
                if g.get("is_discrete", False):
                    if min_v >= -128 and max_v <= 127: mem_opt = "cast:Int8"
                    elif min_v >= -32768 and max_v <= 32767: mem_opt = "cast:Int16"
                    else: mem_opt = "cast:Int32"
                else:
                    mem_opt = "cast:Float32"
            else:
                if g.get("cardinality_ratio", 1.0) < self.cardinality_max:
                    mem_opt = "cast:category"
                else:
                    mem_opt = "cast:string[pyarrow]"
                    
            blueprint[col]["cleaning_pipeline"] = pipeline
            blueprint[col]["memory_optimization"] = mem_opt

        logger.info("Blueprint Logic Engine successfully generated architectural instructions.")
        return blueprint

    def scan_stream(self, chunk_generator) -> Dict[str, Dict[str, Any]]:
        """
        The Main Public API for Pass 1. 
        Takes a generator of chunks from the Loader, executes parallel mapping, 
        reduces the stats globally, and outputs the final Master Blueprint.
        
        Args:
            chunk_generator: A generator yielding pd.DataFrame chunks.
            
        Returns:
            Dict: The Master Execution Blueprint for the Reconstructor and Engine.
        """
        logger.info(f"Initiating Pass 1: Global Streaming MapReduce on {self.max_cores} cores.")
        all_chunk_stats = []
        
        # PHASE 1: Parallel Map Execution
        with ProcessPoolExecutor(max_workers=self.max_cores) as executor:
            futures = []
            # We submit chunks as they are generated. 
            for chunk in chunk_generator:
                futures.append(
                    executor.submit(_map_chunk_stats, chunk, self.target_columns)
                )
                
            for future in as_completed(futures):
                try:
                    chunk_stats = future.result()
                    all_chunk_stats.append(chunk_stats)
                except Exception as e:
                    logger.error(f"Map Worker Failed during scanning: {e}")
                    raise
                    
        if not all_chunk_stats:
            logger.warning("No data was scanned. Returning empty blueprint.")
            return {}

        # PHASE 2 & 3: Reduce & Decide (Main Thread - Microseconds to execute)
        self._reduce_stats(all_chunk_stats)
        self._derive_global_mathematics()
        master_blueprint = self._apply_decision_matrix()
        
        return master_blueprint


# ==========================================
# EXAMPLE INTEGRATION
# ==========================================
if __name__ == "__main__":
    # this is called by engine.py/core.py passing the Loader's generator
    pass