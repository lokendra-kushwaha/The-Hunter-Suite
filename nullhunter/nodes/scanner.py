import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

# ==========================================
# LOGGER CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] NULLHUNTER_BRAIN: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# ISOLATED MAP-WORKER (Micro-API)
# ==========================================
def extract_chunk_statistics(chunk: pd.DataFrame, target_columns: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    The Pure Stateless Map Worker (Phase 1 of MapReduce).
    
    Extracts raw mathematical moments, frequency counts, and text metadata from a single chunk.
    This function is completely decoupled from any multiprocessing logic, making it 100% 
    pickle-safe and universally runnable by any external engine (e.g., ParallelRunner).
    
    Args:
        chunk (pd.DataFrame): A 2D flat slice of the dataset.
        target_columns (Optional[List[str]]): Specific columns to profile. If None, profiles all.
        
    Returns:
        Dict[str, Dict[str, Any]]: Raw chunk statistics mapped by column name.
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
        
        # Base state dictionary
        stats: Dict[str, Any] = {
            "total_rows": total_rows,
            "missing_count": missing_count,
            "valid_count": valid_count,
            "is_numeric": is_numeric,
            "sum_x": 0.0,
            "sum_x2": 0.0,
            "sum_x3": 0.0,
            "sum_x4": 0.0,  # Required for Kurtosis calculation
            "zero_count": 0,
            "negative_count": 0,
            "min": float('inf'),
            "max": float('-inf'),
            "unique_sample": set(),
            "sum_text_len": 0,
            "max_text_len": 0
        }

        if valid_count > 0:
            if is_numeric:
                # Core mathematical moments for Mean, Variance, Skewness, and Kurtosis
                stats["sum_x"] = float(np.sum(valid_data))
                stats["sum_x2"] = float(np.sum(valid_data ** 2))
                stats["sum_x3"] = float(np.sum(valid_data ** 3))
                stats["sum_x4"] = float(np.sum(valid_data ** 4))
                
                stats["min"] = float(np.min(valid_data))
                stats["max"] = float(np.max(valid_data))
                stats["zero_count"] = int((valid_data == 0).sum())
                stats["negative_count"] = int((valid_data < 0).sum())
                
                # Integer checking for Predictive Mean Matching (PMM)
                stats["is_discrete"] = bool(np.all(np.mod(valid_data, 1) == 0))
            else:
                # Capture text length metrics for NLP Detection
                text_data = valid_data.astype(str)
                text_lengths = text_data.str.len()
                stats["sum_text_len"] = int(text_lengths.sum())
                stats["max_text_len"] = int(text_lengths.max())

                # Capture a capped set of unique values to prevent RAM explosion
                unique_vals = valid_data.unique()
                if len(unique_vals) <= 25000:
                    stats["unique_sample"] = set(unique_vals)
                else:
                    stats["unique_sample"] = set(unique_vals[:25000])
                    stats["high_cardinality_flag"] = True

        chunk_stats[col] = stats
        
    return chunk_stats


# ==========================================
# THE SCANNER CLASS (The Brain)
# ==========================================
class DataScanner:
    """
    The Autonomous Statistical Profiler & AI Blueprint Generator for NullHunter.
    
    This module strictly adheres to the Single Responsibility Principle (SRP). 
    It manages NO CPU cores and NO execution pipelines. It acts solely as the 
    'Diagnostic Physician'—reducing mapped stats and generating a deterministic JSON Blueprint.
    """

    def __init__(
        self,
        # Sparsity & Core Thresholds
        drop_threshold: float = 0.85,
        constant_variance_drop: bool = True,
        
        # Distribution & Outlier Thresholds
        skewness_high_threshold: float = 1.5,
        skewness_extreme_threshold: float = 3.0,
        kurtosis_threshold: float = 3.0,
        outlier_z_threshold: float = 3.5,
        zero_inflation_trigger: float = 0.30,
        
        # Advanced ML Algorithm Triggers
        svd_sparsity_trigger: float = 0.40,
        mice_correlation_min: float = 0.10,
        cardinality_ratio_max: float = 0.05,
        nlp_text_length_trigger: float = 40.0,
        
        # Execution Modes
        time_series_mode: bool = False,
        deep_learning_mode: bool = True
    ) -> None:
        """
        Initializes the Scanner with an extensive array of enterprise heuristic parameters.
        
        Args:
            drop_threshold (float): % of NaNs above which column is dropped.
            constant_variance_drop (bool): Drop columns where all values are identical.
            skewness_high_threshold (float): Threshold to shift from Mean to Median/KNN.
            skewness_extreme_threshold (float): Threshold to trigger transformations (Yeo-Johnson).
            kurtosis_threshold (float): Threshold indicating heavy-tailed distributions (Outlier swarms).
            outlier_z_threshold (float): Standard deviations to define an extreme outlier.
            zero_inflation_trigger (float): % of zeros triggering Zero-Inflated imputation layers.
            svd_sparsity_trigger (float): Missing % threshold to recommend Matrix Factorization (SVD).
            mice_correlation_min (float): Minimum missing % to justify heavy MICE iteration.
            cardinality_ratio_max (float): Max unique/total ratio for categorical variables.
            nlp_text_length_trigger (float): Average string length to classify column as NLP Free-Text.
            time_series_mode (bool): If True, biases towards Spline/Kalman interpolation.
            deep_learning_mode (bool): If True, allows Neural Autoencoder recommendations.
        """
        self.drop_threshold = drop_threshold
        self.constant_variance_drop = constant_variance_drop
        self.skew_high = skewness_high_threshold
        self.skew_extreme = skewness_extreme_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_z = outlier_z_threshold
        self.zero_inflation_trigger = zero_inflation_trigger
        
        self.svd_trigger = svd_sparsity_trigger
        self.mice_trigger = mice_correlation_min
        self.cardinality_max = cardinality_ratio_max
        self.nlp_trigger = nlp_text_length_trigger
        
        self.time_series_mode = time_series_mode
        self.deep_learning_mode = deep_learning_mode
        
        self.global_stats: Dict[str, Dict[str, Any]] = {}
        logger.info("Scanner Initialization Complete. Ready to process reduced mathematical streams.")

    def reduce_stats(self, all_chunk_stats: List[Dict[str, Dict[str, Any]]]) -> None:
        """
        The Master Reducer (Phase 2 of MapReduce).
        Merges a list of intermediate chunk statistics into a single Global State Matrix.
        
        Args:
            all_chunk_stats (List[Dict]): A list containing the outputs of `extract_chunk_statistics`.
        """
        logger.debug("Reducing chunk statistics into Global State Matrices...")
        
        for chunk in all_chunk_stats:
            for col, stats in chunk.items():
                if col not in self.global_stats:
                    # Initialize global aggregator for this column
                    self.global_stats[col] = {
                        "total_rows": 0, "missing_count": 0, "valid_count": 0,
                        "is_numeric": stats["is_numeric"], "is_discrete": stats.get("is_discrete", False),
                        "sum_x": 0.0, "sum_x2": 0.0, "sum_x3": 0.0, "sum_x4": 0.0,
                        "zero_count": 0, "negative_count": 0,
                        "min": float('inf'), "max": float('-inf'),
                        "unique_set": set(), "high_cardinality_flag": False,
                        "sum_text_len": 0, "max_text_len": 0
                    }
                
                g = self.global_stats[col]
                g["total_rows"] += stats["total_rows"]
                g["missing_count"] += stats["missing_count"]
                g["valid_count"] += stats["valid_count"]
                
                if stats["is_numeric"]:
                    g["sum_x"] += stats["sum_x"]
                    g["sum_x2"] += stats["sum_x2"]
                    g["sum_x3"] += stats["sum_x3"]
                    g["sum_x4"] += stats["sum_x4"]
                    g["zero_count"] += stats["zero_count"]
                    g["negative_count"] += stats["negative_count"]
                    g["min"] = min(g["min"], stats["min"])
                    g["max"] = max(g["max"], stats["max"])
                    
                    if not stats.get("is_discrete", True):
                        g["is_discrete"] = False
                else:
                    g["sum_text_len"] += stats["sum_text_len"]
                    g["max_text_len"] = max(g["max_text_len"], stats["max_text_len"])
                    
                    if not g["high_cardinality_flag"]:
                        g["unique_set"].update(stats.get("unique_sample", set()))
                        if len(g["unique_set"]) > 100000 or stats.get("high_cardinality_flag", False):
                            g["high_cardinality_flag"] = True
                            g["unique_set"].clear() # Free RAM

    def _derive_global_mathematics(self) -> None:
        """
        Derives high-level Population Mathematics (Variance, Skewness, Kurtosis) 
        from the raw central moments using advanced statistical formulas.
        """
        for col, g in self.global_stats.items():
            g["missing_ratio"] = g["missing_count"] / g["total_rows"] if g["total_rows"] > 0 else 1.0
            
            if g["is_numeric"] and g["valid_count"] > 3:
                N = g["valid_count"]
                mean = g["sum_x"] / N
                
                # Variance (M2)
                variance = max(0.0, (g["sum_x2"] / N) - (mean ** 2))
                std_dev = math.sqrt(variance)
                
                # Higher Order Moments (Skewness & Kurtosis)
                if std_dev > 0:
                    # M3 (Third Central Moment) for Skewness
                    m3 = (g["sum_x3"] / N) - 3 * mean * (g["sum_x2"] / N) + 2 * (mean ** 3)
                    skewness = m3 / (std_dev ** 3)
                    
                    # M4 (Fourth Central Moment) for Kurtosis (Pearson's)
                    m4 = (g["sum_x4"] / N) - 4 * mean * (g["sum_x3"] / N) + 6 * (mean ** 2) * (g["sum_x2"] / N) - 3 * (mean ** 4)
                    kurtosis = m4 / (std_dev ** 4)
                else:
                    skewness = 0.0
                    kurtosis = 0.0
                    
                g["global_mean"] = mean
                g["global_std"] = std_dev
                g["global_skewness"] = skewness
                g["global_kurtosis"] = kurtosis
                g["zero_ratio"] = g["zero_count"] / N
                
            elif not g["is_numeric"] and g["valid_count"] > 0:
                g["cardinality_ratio"] = len(g["unique_set"]) / g["valid_count"] if not g["high_cardinality_flag"] else 1.0
                g["avg_text_len"] = g["sum_text_len"] / g["valid_count"]

    def generate_blueprint(self) -> Dict[str, Dict[str, Any]]:
        """
        The Brain Engine (Phase 3). Evaluates the derived Population Mathematics 
        against a strict AI Heuristic Ruleset to construct the Final Master Blueprint.
        
        Returns:
            Dict[str, Dict[str, Any]]: The finalized Master Execution Blueprint.
        """
        self._derive_global_mathematics()
        blueprint: Dict[str, Dict[str, Any]] = {}
        
        for col, g in self.global_stats.items():
            pipeline = []
            mem_opt = None
            missing_ratio = g["missing_ratio"]
            
            blueprint[col] = {"cleaning_pipeline": pipeline, "memory_optimization": mem_opt}
            
            # --- ABSOLUTE SPARSITY & HYGIENE RULES ---
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
            if g["is_numeric"]:
                skew = abs(g.get("global_skewness", 0.0))
                kurtosis = g.get("global_kurtosis", 0.0)
                is_discrete = g.get("is_discrete", False)
                
                # 1. Distribution Shape & Outlier Management
                mean, std = g.get("global_mean", 0), g.get("global_std", 1)
                has_extreme_outliers = (abs(g["max"] - mean) / std > self.outlier_z) or (abs(g["min"] - mean) / std > self.outlier_z)

                if has_extreme_outliers or kurtosis > self.kurtosis_threshold:
                    if g["negative_count"] > 0:
                        pipeline.append("yeo_johnson_transformer") # Handles negative skewed data
                    else:
                        pipeline.append("isolation_forest_outlier_remover") # Advanced ML outlier detection
                elif skew > self.skew_extreme:
                    pipeline.append("box_cox_transformer") # Standard power transform for strictly positive data

                # 2. Missing Value Imputation Algorithms
                if missing_ratio > 0:
                    if self.time_series_mode:
                        pipeline.append("spline_interpolation" if not is_discrete else "ffill_bfill_imputer")
                        
                    elif g.get("zero_ratio", 0.0) > self.zero_inflation_trigger:
                        pipeline.append("zero_inflated_poisson_imputer") # Specialized for heavy-zero data
                        
                    elif missing_ratio > self.svd_trigger:
                        if self.deep_learning_mode and not is_discrete:
                            pipeline.append("autoencoder_deep_imputer")
                        else:
                            pipeline.append("svd_matrix_factorization")
                            
                    elif missing_ratio > self.mice_trigger:
                        if skew > self.skew_extreme:
                            pipeline.append("mahalanobis_knn_imputer") 
                        elif is_discrete:
                            pipeline.append("pmm_imputer") 
                        else:
                            pipeline.append("mice_iterative_imputer") 
                            
                    else:
                        if skew > self.skew_high:
                            pipeline.append("knn_imputer" if missing_ratio > 0.02 else "median_imputer")
                        else:
                            pipeline.append("em_algorithm_imputer" if missing_ratio > 0.01 else "mean_imputer")
            
            # --- CATEGORICAL & NLP TEXT IMPUTATION LOGIC ---
            elif not g["is_numeric"]:
                avg_len = g.get("avg_text_len", 0.0)
                
                # Detect if this is Free-Text (NLP) or just Categorical
                if avg_len >= self.nlp_trigger:
                    pipeline.append("nlp_text_cleaner")
                    if missing_ratio > 0:
                        pipeline.append("empty_string_imputer")
                else:
                    if missing_ratio > 0:
                        cardinality = g.get("cardinality_ratio", 1.0)
                        if cardinality < self.cardinality_max:
                            pipeline.append("mode_imputer")
                        elif self.deep_learning_mode:
                            pipeline.append("missforest_categorical_imputer")
                        else:
                            pipeline.append("flag_missing_text")
                    
                    pipeline.append("regex_whitespace_cleaner")

            # --- MEMORY OPTIMIZATION DECISIONS ---
            if g["is_numeric"]:
                min_v, max_v = g["min"], g["max"]
                if g.get("is_discrete", False) and g["missing_count"] == 0:
                    # Strict integer casting if no NaNs (Pandas Int8 supports NaNs, but standard int doesn't)
                    if min_v >= -128 and max_v <= 127: mem_opt = "cast:Int8"
                    elif min_v >= -32768 and max_v <= 32767: mem_opt = "cast:Int16"
                    else: mem_opt = "cast:Int32"
                else:
                    mem_opt = "cast:Float32"
            else:
                if g.get("cardinality_ratio", 1.0) < self.cardinality_max:
                    mem_opt = "cast:category"
                elif g.get("avg_text_len", 0.0) < self.nlp_trigger:
                    mem_opt = "cast:string[pyarrow]"
                    
            blueprint[col]["cleaning_pipeline"] = pipeline
            blueprint[col]["memory_optimization"] = mem_opt

        logger.info("Blueprint Logic Engine successfully generated advanced architectural instructions.")
        return blueprint

    