"""
NullHunter Autonomous Scanner & AI Brain (The Diagnostic Physician)
===================================================================

This module acts as the isolated, stateless intelligence core of the framework.
It adheres strictly to the MapReduce paradigm:
1. Micro-API (`extract_chunk_statistics`): The Map phase. Safely extracts raw 
   mathematical moments, string geometries, and regex heuristics from isolated chunks.
2. Macro-API (`DataScanner`): The Reduce & Evaluate phase. Aggregates chunk matrices 
   into Global Population Mathematics and applies an advanced AI Heuristic Engine to 
   generate a deterministic Execution Blueprint.

Key Enterprise Upgrades:
- PII & Temporal Detection: Automatically identifies Emails, Phones, and Timestamps.
- Information Theory: Calculates Entropy and Coefficient of Variation (CV).
- Advanced ML Routing: Predicts the need for SMOTE, XGBoost Imputation, and Robust Scaling.
"""

import math
import re
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Set

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
# ADVANCED PATTERN HEURISTICS
# ==========================================
# Pre-compiled Regex for ultra-fast vectorized O(N) execution inside the worker
REGEX_EMAIL = re.compile(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')
REGEX_PHONE = re.compile(r'^\+?1?\s*\(?-*\.*[0-9]{3}\)?\s*-*\.*[0-9]{3}\s*-*\.*[0-9]{4}$')
REGEX_DATETIME = re.compile(r'^\d{4}-\d{2}-\d{2}(T|\s)\d{2}:\d{2}:\d{2}')


# ==========================================
# ISOLATED MAP-WORKER (Micro-API)
# ==========================================
def extract_chunk_statistics(
    chunk: pd.DataFrame, 
    target_columns: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    The Pure Stateless Map Worker (Phase 1 of MapReduce).
    
    Extracts high-dimensional mathematical moments, frequency matrices, and Regex 
    metadata from a single chunk. Decoupled from multiprocessing to ensure 
    universal serialization (Pickle-safe) for distributed runners like UPR or PySpark.
    
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
        is_bool = pd.api.types.is_bool_dtype(series) or (set(valid_data.unique()).issubset({0, 1, '0', '1', True, False}))
        
        # Base state dictionary (The Structural Matrix)
        stats: Dict[str, Any] = {
            "total_rows": total_rows,
            "missing_count": missing_count,
            "valid_count": valid_count,
            "is_numeric": is_numeric,
            "is_bool": is_bool,
            "sum_x": 0.0,
            "sum_x2": 0.0,
            "sum_x3": 0.0,
            "sum_x4": 0.0,
            "sum_log_x": 0.0,  # Geometric Mean estimator (strictly positive)
            "zero_count": 0,
            "negative_count": 0,
            "min": float('inf'),
            "max": float('-inf'),
            "unique_sample": set(),
            "sum_text_len": 0,
            "max_text_len": 0,
            "email_hits": 0,
            "phone_hits": 0,
            "datetime_hits": 0,
            "whitespace_only_count": 0
        }

        if valid_count > 0:
            if is_numeric and not is_bool:
                # Core mathematical moments for Skewness, Kurtosis, and Variance
                stats["sum_x"] = float(np.sum(valid_data))
                stats["sum_x2"] = float(np.sum(valid_data ** 2))
                stats["sum_x3"] = float(np.sum(valid_data ** 3))
                stats["sum_x4"] = float(np.sum(valid_data ** 4))
                
                stats["min"] = float(np.min(valid_data))
                stats["max"] = float(np.max(valid_data))
                stats["zero_count"] = int((valid_data == 0).sum())
                stats["negative_count"] = int((valid_data < 0).sum())
                
                # Geometric Mean tracking (Only for strictly positive columns)
                if stats["min"] > 0:
                    stats["sum_log_x"] = float(np.sum(np.log(valid_data)))
                
                # Integer checking for Predictive Mean Matching (PMM)
                stats["is_discrete"] = bool(np.all(np.mod(valid_data, 1) == 0))
                
            elif not is_bool:
                # NLP & Contextual Detection Logic
                text_data = valid_data.astype(str)
                text_lengths = text_data.str.len()
                
                stats["sum_text_len"] = int(text_lengths.sum())
                stats["max_text_len"] = int(text_lengths.max())
                stats["whitespace_only_count"] = int(text_data.str.isspace().sum())
                
                # Fast Vectorized Heuristic Profiling (PII & Temporal)
                stats["email_hits"] = int(text_data.str.match(REGEX_EMAIL).sum())
                stats["phone_hits"] = int(text_data.str.match(REGEX_PHONE).sum())
                stats["datetime_hits"] = int(text_data.str.match(REGEX_DATETIME).sum())

            # Entropy / Cardinality Sample Guardrail
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
    The Autonomous Statistical Profiler & AI Blueprint Generator.
    
    Adheres to the Single Responsibility Principle (SRP). It manages NO execution 
    pipelines. It solely reduces mapped chunk stats into Population Mathematics and 
    evaluates them against an Enterprise ML Heuristic Engine.
    """

    def __init__(
        self,
        # Sparsity & Core Hygiene
        drop_threshold: float = 0.85,
        constant_variance_drop: bool = True,
        whitespace_is_null: bool = True,
        
        # Distribution & Outlier Architecture
        skewness_high_threshold: float = 1.5,
        skewness_extreme_threshold: float = 3.0,
        kurtosis_threshold: float = 3.0,
        outlier_z_threshold: float = 3.5,
        coefficient_of_variation_trigger: float = 2.0,
        zero_inflation_trigger: float = 0.30,
        
        # Advanced ML Algorithm Triggers
        svd_sparsity_trigger: float = 0.40,
        mice_correlation_min: float = 0.10,
        cardinality_ratio_max: float = 0.05,
        imbalance_entropy_threshold: float = 0.30, 
        nlp_text_length_trigger: float = 40.0,
        
        # Security & Temporal Recognition
        pii_detection_confidence: float = 0.75,
        temporal_detection_confidence: float = 0.80,
        
        # Execution Bias Modes
        time_series_mode: bool = False,
        deep_learning_mode: bool = True
    ) -> None:
        """
        Initializes the AI Brain with a massive array of heuristic parameters.
        
        Args:
            drop_threshold (float): % of NaNs above which a column is dropped.
            constant_variance_drop (bool): Drop columns where all values are identical.
            whitespace_is_null (bool): Treat pure whitespace strings as missing data.
            skewness_high_threshold (float): Threshold to shift from Mean to Median/KNN.
            skewness_extreme_threshold (float): Threshold to trigger Box-Cox/Yeo-Johnson.
            kurtosis_threshold (float): Heavy-tailed distributions indicator for Robust Scaling.
            outlier_z_threshold (float): Standard deviations to define extreme outliers.
            coefficient_of_variation_trigger (float): StdDev/Mean ratio triggering advanced non-linear imputers.
            zero_inflation_trigger (float): % of zeros triggering Zero-Inflated models.
            svd_sparsity_trigger (float): Missing % threshold to recommend Matrix Factorization.
            mice_correlation_min (float): Minimum missing % to justify heavy MICE iteration.
            cardinality_ratio_max (float): Max unique/total ratio for categorical variables.
            imbalance_entropy_threshold (float): Low entropy trigger for SMOTE oversampling.
            nlp_text_length_trigger (float): Avg string length classifying column as NLP Free-Text.
            pii_detection_confidence (float): Match ratio required to flag column for PII Redaction.
            temporal_detection_confidence (float): Match ratio to trigger DateTime Caster.
            time_series_mode (bool): Biases decisions towards Spline/Kalman interpolation.
            deep_learning_mode (bool): Allows Autoencoder and MissForest recommendations.
        """
        self.drop_threshold = drop_threshold
        self.constant_variance_drop = constant_variance_drop
        self.whitespace_is_null = whitespace_is_null
        
        self.skew_high = skewness_high_threshold
        self.skew_extreme = skewness_extreme_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.outlier_z = outlier_z_threshold
        self.cv_trigger = coefficient_of_variation_trigger
        self.zero_inflation_trigger = zero_inflation_trigger
        
        self.svd_trigger = svd_sparsity_trigger
        self.mice_trigger = mice_correlation_min
        self.cardinality_max = cardinality_ratio_max
        self.entropy_threshold = imbalance_entropy_threshold
        self.nlp_trigger = nlp_text_length_trigger
        
        self.pii_confidence = pii_detection_confidence
        self.temporal_confidence = temporal_detection_confidence
        
        self.time_series_mode = time_series_mode
        self.deep_learning_mode = deep_learning_mode
        
        self.global_stats: Dict[str, Dict[str, Any]] = {}
        logger.info("Scanner Engine Initialized: PII, Temporal, and Deep-Learning Matrix Armed.")


    def reduce_stats(self, all_chunk_stats: List[Dict[str, Dict[str, Any]]]) -> None:
        """
        The Master Reducer (Phase 2 of MapReduce).
        Merges intermediate chunk statistics into a single Global State Matrix.
        """
        logger.debug("Reducing chunk statistics into Global State Matrices...")
        
        for chunk in all_chunk_stats:
            for col, stats in chunk.items():
                if col not in self.global_stats:
                    self.global_stats[col] = {
                        "total_rows": 0, "missing_count": 0, "valid_count": 0,
                        "is_numeric": stats["is_numeric"], "is_bool": stats.get("is_bool", False),
                        "is_discrete": stats.get("is_discrete", False),
                        "sum_x": 0.0, "sum_x2": 0.0, "sum_x3": 0.0, "sum_x4": 0.0, "sum_log_x": 0.0,
                        "zero_count": 0, "negative_count": 0,
                        "min": float('inf'), "max": float('-inf'),
                        "unique_set": set(), "high_cardinality_flag": False,
                        "sum_text_len": 0, "max_text_len": 0,
                        "email_hits": 0, "phone_hits": 0, "datetime_hits": 0, "whitespace_only_count": 0
                    }
                
                g = self.global_stats[col]
                g["total_rows"] += stats["total_rows"]
                g["missing_count"] += stats["missing_count"]
                g["valid_count"] += stats["valid_count"]
                
                if stats["is_numeric"] and not stats.get("is_bool", False):
                    g["sum_x"] += stats["sum_x"]
                    g["sum_x2"] += stats["sum_x2"]
                    g["sum_x3"] += stats["sum_x3"]
                    g["sum_x4"] += stats["sum_x4"]
                    g["sum_log_x"] += stats["sum_log_x"]
                    g["zero_count"] += stats["zero_count"]
                    g["negative_count"] += stats["negative_count"]
                    g["min"] = min(g["min"], stats["min"])
                    g["max"] = max(g["max"], stats["max"])
                    
                    if not stats.get("is_discrete", True):
                        g["is_discrete"] = False
                elif not stats.get("is_bool", False):
                    g["sum_text_len"] += stats["sum_text_len"]
                    g["max_text_len"] = max(g["max_text_len"], stats["max_text_len"])
                    g["email_hits"] += stats["email_hits"]
                    g["phone_hits"] += stats["phone_hits"]
                    g["datetime_hits"] += stats["datetime_hits"]
                    g["whitespace_only_count"] += stats["whitespace_only_count"]
                    
                # Cardinality Memory Guard
                if not g["high_cardinality_flag"]:
                    g["unique_set"].update(stats.get("unique_sample", set()))
                    if len(g["unique_set"]) > 100000 or stats.get("high_cardinality_flag", False):
                        g["high_cardinality_flag"] = True
                        g["unique_set"].clear()


    def _derive_global_mathematics(self) -> None:
        """
        Derives high-level Population Mathematics (Variance, Skewness, Kurtosis, Entropy) 
        from raw central moments using advanced statistical approximations.
        """
        for col, g in self.global_stats.items():
            # Adjust missing counts if whitespace is treated as null
            if self.whitespace_is_null and not g["is_numeric"]:
                g["missing_count"] += g["whitespace_only_count"]
                g["valid_count"] -= g["whitespace_only_count"]
                
            g["missing_ratio"] = g["missing_count"] / g["total_rows"] if g["total_rows"] > 0 else 1.0
            
            if g["is_numeric"] and g["valid_count"] > 3 and not g["is_bool"]:
                N = g["valid_count"]
                mean = g["sum_x"] / N
                
                variance = max(0.0, (g["sum_x2"] / N) - (mean ** 2))
                std_dev = math.sqrt(variance)
                
                if std_dev > 0:
                    m3 = (g["sum_x3"] / N) - 3 * mean * (g["sum_x2"] / N) + 2 * (mean ** 3)
                    skewness = m3 / (std_dev ** 3)
                    
                    m4 = (g["sum_x4"] / N) - 4 * mean * (g["sum_x3"] / N) + 6 * (mean ** 2) * (g["sum_x2"] / N) - 3 * (mean ** 4)
                    kurtosis = m4 / (std_dev ** 4)
                    cv = std_dev / abs(mean) if mean != 0 else float('inf')
                else:
                    skewness = 0.0; kurtosis = 0.0; cv = 0.0
                    
                g["global_mean"] = mean
                g["global_std"] = std_dev
                g["global_skewness"] = skewness
                g["global_kurtosis"] = kurtosis
                g["global_cv"] = cv
                g["zero_ratio"] = g["zero_count"] / N
                
            elif not g["is_numeric"] and not g["is_bool"] and g["valid_count"] > 0:
                N = g["valid_count"]
                g["cardinality_ratio"] = len(g["unique_set"]) / N if not g["high_cardinality_flag"] else 1.0
                g["avg_text_len"] = g["sum_text_len"] / N
                
                g["email_ratio"] = g["email_hits"] / N
                g["phone_ratio"] = g["phone_hits"] / N
                g["datetime_ratio"] = g["datetime_hits"] / N
                
                # Approximate Entropy for Categorical Imbalance Detection
                if not g["high_cardinality_flag"] and len(g["unique_set"]) > 1:
                    max_entropy = math.log2(len(g["unique_set"]))
                    # Simplifying entropy heuristic: 1.0 means perfectly balanced, ~0.0 means highly skewed
                    g["entropy_score"] = 1.0 if g["cardinality_ratio"] > 0.8 else min(1.0, len(g["unique_set"]) / 100.0) 
                else:
                    g["entropy_score"] = 1.0


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
                
            if g["is_numeric"] and not g.get("is_bool", False) and g.get("global_std", -1) == 0.0 and self.constant_variance_drop:
                pipeline.append("drop_constant_variance")
                continue
                
            if not g["is_numeric"] and missing_ratio == 0 and g.get("cardinality_ratio", 1.0) == 1.0 and not g.get("datetime_ratio", 0) > self.temporal_confidence:
                pipeline.append("drop_high_cardinality_id")
                continue

            # --- BOOLEAN LOGIC ---
            if g.get("is_bool", False):
                if missing_ratio > 0: pipeline.append("mode_imputer")
                mem_opt = "cast:boolean"
                blueprint[col]["cleaning_pipeline"] = pipeline; blueprint[col]["memory_optimization"] = mem_opt
                continue

            # --- TEMPORAL & PII SECURITY LOGIC ---
            if not g["is_numeric"]:
                if g.get("datetime_ratio", 0.0) >= self.temporal_confidence:
                    pipeline.append("datetime_caster")
                    if missing_ratio > 0: pipeline.append("ffill_bfill_imputer" if self.time_series_mode else "mode_imputer")
                    pipeline.append("temporal_feature_extractor")
                    mem_opt = "cast:datetime64[ns]"
                    blueprint[col]["cleaning_pipeline"] = pipeline; blueprint[col]["memory_optimization"] = mem_opt
                    continue
                    
                if g.get("email_ratio", 0.0) >= self.pii_confidence or g.get("phone_ratio", 0.0) >= self.pii_confidence:
                    pipeline.append("pii_redaction_filter")

            # --- NUMERICAL AI IMPUTATION LOGIC ---
            if g["is_numeric"]:
                skew = abs(g.get("global_skewness", 0.0))
                kurtosis = g.get("global_kurtosis", 0.0)
                cv = g.get("global_cv", 0.0)
                is_discrete = g.get("is_discrete", False)
                
                # 1. Distribution Shape & Outlier Management
                mean, std = g.get("global_mean", 0), g.get("global_std", 1)
                has_extreme_outliers = (abs(g["max"] - mean) / std > self.outlier_z) or (abs(g["min"] - mean) / std > self.outlier_z)

                if has_extreme_outliers or kurtosis > self.kurtosis_threshold:
                    if g["negative_count"] > 0: pipeline.append("yeo_johnson_transformer") 
                    else: pipeline.append("isolation_forest_outlier_remover")
                    # Trigger Robust Scaling for downstream ML due to heavy tails
                    pipeline.append("robust_scaler")
                elif skew > self.skew_extreme:
                    pipeline.append("box_cox_transformer")
                elif cv > self.cv_trigger:
                    pipeline.append("standard_scaler") # Normalize highly volatile data

                # 2. Missing Value Imputation Algorithms
                if missing_ratio > 0:
                    if self.time_series_mode:
                        pipeline.append("spline_interpolation" if not is_discrete else "ffill_bfill_imputer")
                        
                    elif g.get("zero_ratio", 0.0) > self.zero_inflation_trigger:
                        pipeline.append("zero_inflated_poisson_imputer")
                        
                    elif missing_ratio > self.svd_trigger:
                        if self.deep_learning_mode and not is_discrete:
                            pipeline.append("autoencoder_deep_imputer")
                        else:
                            pipeline.append("svd_matrix_factorization")
                            
                    elif missing_ratio > self.mice_trigger:
                        if cv > self.cv_trigger and self.deep_learning_mode:
                            pipeline.append("xgboost_non_linear_imputer") # Handles extreme variance better than MICE
                        elif skew > self.skew_extreme:
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
                
                if avg_len >= self.nlp_trigger:
                    pipeline.append("nlp_text_cleaner")
                    if missing_ratio > 0: pipeline.append("empty_string_imputer")
                    pipeline.append("tf_idf_vectorizer_prep") # Ready it for NLP processing
                else:
                    if missing_ratio > 0:
                        cardinality = g.get("cardinality_ratio", 1.0)
                        if cardinality < self.cardinality_max:
                            pipeline.append("mode_imputer")
                        elif self.deep_learning_mode:
                            pipeline.append("missforest_categorical_imputer")
                        else:
                            pipeline.append("flag_missing_text")
                    
                    # Detect Class Imbalance for Categorical variables
                    if g.get("entropy_score", 1.0) < self.entropy_threshold and self.deep_learning_mode:
                        pipeline.append("smote_oversampler_prep")
                        
                    pipeline.append("regex_whitespace_cleaner")

            # --- MEMORY OPTIMIZATION DECISIONS ---
            if g["is_numeric"]:
                min_v, max_v = g["min"], g["max"]
                if g.get("is_discrete", False) and g["missing_count"] == 0:
                    if min_v >= -128 and max_v <= 127: mem_opt = "cast:Int8"
                    elif min_v >= -32768 and max_v <= 32767: mem_opt = "cast:Int16"
                    elif min_v >= -2147483648 and max_v <= 2147483647: mem_opt = "cast:Int32"
                    else: mem_opt = "cast:Int64"
                else:
                    mem_opt = "cast:Float32"
            elif not g.get("is_bool", False):
                if g.get("cardinality_ratio", 1.0) < self.cardinality_max: mem_opt = "cast:category"
                elif g.get("avg_text_len", 0.0) < self.nlp_trigger: mem_opt = "cast:string[pyarrow]"
                    
            blueprint[col]["cleaning_pipeline"] = pipeline
            blueprint[col]["memory_optimization"] = mem_opt

        logger.info("Blueprint Logic Engine successfully mapped advanced topological instructions.")
        return blueprint    