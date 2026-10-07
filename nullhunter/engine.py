import os
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Union, List, Dict, Optional

class _ExecutionEngine:
    """
    The Master Orchestrator (The Muscle & Brain).
    Handles smart multiprocessing dispatch, blueprint validation, and API routing.
    """
    def __init__(self, loader_ref):
        self.loader = loader_ref
        
        # Registry of all valid cleaning layers/operations for validation
        self.valid_layers = {
            'type_caster', 'anomaly_hunter', 'outlier_capper', 
            'time_series_ffill', 'group_mean_imputer', 'knn_imputer', 'linear_predictor'
        }

    def _validate_blueprint(self, blueprint: Dict) -> bool:
        """
        Prevents downstream runtime crashes by validating user-defined rules 
        before a single row of data is mutated.
        """
        print("[NullHunter Validator] Inspecting user action blueprint...")
        for col, operations in blueprint.items():
            if not isinstance(operations, list):
                raise TypeError(f"[NullHunter Error] Column '{col}' operations must be passed inside a List (e.g., ['ffill']).")
            
            for op in operations:
                if op not in self.valid_layers:
                    raise ValueError(
                        f"[NullHunter Error] Invalid cleaning operation '{op}' specified for column '{col}'. "
                        f"Allowed operations are: {list(self.valid_layers)}"
                    )
        print("[NullHunter Validator] Blueprint validation passed successfully.")
        return True

    def _smart_dispatch(self, chunks_iterable) -> List:
        """
        The Smart Dispatcher: Dynamically decides whether to use multi-core 
        parallelism or single-threaded execution based on dataset size and CPU cores.
        """
        # Convert generator to list to inspect total chunks
        chunks = list(chunks_iterable)
        total_chunks = len(chunks)
        
        # If dataset is too small, multiprocessing is a waste of CPU spin-up time
        if total_chunks <= 1:
            print("[NullHunter Dispatcher] Small dataset detected. Executing on Single Thread (Zero Overhead).")
            return [self._process_chunk(chunk) for chunk in chunks]
            
        # Get available physical CPU cores
        available_cores = os.cpu_count() or 2
        active_workers = min(available_cores, total_chunks)
        
        print(f"[NullHunter Dispatcher] Heavy dataset detected. Deploying Beast Mode across {active_workers} CPU Cores...")
        
        processed_results = []
        with ProcessPoolExecutor(max_workers=active_workers) as executor:
            # Map chunks concurrently across physical cores
            futures = {executor.submit(self._process_chunk, chunk): i for i, chunk in enumerate(chunks)}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    processed_results.append(result)
                except Exception as e:
                    print(f"[NullHunter Error] Multiprocessing core crashed: {e}")
                    raise e
                    
        return processed_results

    def _process_chunk(self, chunk):
        """Worker function executed by individual CPU cores."""
        # Placeholder for scanning/cleaning logic routing
        return chunk

    def scan(self, 
             data: object, 
             target_columns: Optional[List[str]] = None, 
             deep_scan: bool = True,
             outlier_threshold: float = 3.0,
             n_jobs: int = -1) -> Dict:
        """The X-Ray Scanner. Generates a JSON-based mathematical Action Plan."""
        print("[NullHunter Scanner] Initializing dataset profiling...")
        # TODO: Route to scanner.py logic
        return {"status": "scanned", "recommendations": {}}

    def purify(self, 
               data: object, 
               action_plan: Dict, 
               inplace: bool = False,
               fallback_strategy: str = 'median') -> object:
        """The Worker Delegator. Routes data to specialized cleaning nodes."""
        print("[NullHunter Purifier] Executing mutation plan across data matrix...")
        # TODO: Route to layers/ modules
        return data

    def how_to_use(self):
        """Prints the complete VIP Developer Protocol and usage manual."""
        manual = """
        ============================================================
        🎯 THE NULLHUNTER PIPELINE - DEVELOPER MANUAL
        ============================================================
        Welcome to the AI-powered Data Cleansing Matrix.

        1. AUTO-PILOT MODE (Zero Effort)
           >>> import nullhunter
           >>> nh = nullhunter.NullHunter()
           >>> clean_df = nh.execute("messy_data.csv", mode='auto')

        2. COPILOT MODE (Hybrid AI + Custom Rules)
           >>> my_rules = {'Salary': ['group_mean_imputer']}
           >>> clean_df = nh.execute("data.csv", mode='copilot', blueprint=my_rules)

        3. STRICT MODE (100% User Control)
           >>> clean_df = nh.execute("data.csv", mode='strict', blueprint=my_rules)

        4. GRANULAR CONTROL (Step-by-Step)
           >>> loader_obj = nh.dataloader(filepath="data.csv", chunk_size=50000)
           >>> chunks = loader_obj.get_chunks()
           >>> plan = nh.execute.scan(chunks, deep_scan=True)
           >>> clean_df = nh.execute.purify(chunks, plan)
        ============================================================
        """
        print(manual)

    def __call__(self, 
                 raw_data: str, 
                 mode: str = 'auto', 
                 blueprint: Optional[Dict] = None,
                 return_report: bool = True,
                 n_jobs: int = -1) -> object:
        """The VIP 'All-In-One' Execution Protocol."""
        print(f"\n[NullHunter] Initializing Pipeline in '{mode.upper()}' mode...")
        
        # 1. Initialize Loader and fetch chunk generator
        active_loader = self.loader(filepath=raw_data)
        chunks_iterable = active_loader.get_chunks()
        
        # 2. Execution Routing based on Mode
        if mode == 'strict' and blueprint:
            self._validate_blueprint(blueprint)
            print("[NullHunter] Strict Mode: Honoring user blueprint exclusively.")
            clean_data = self.purify(data=chunks_iterable, action_plan=blueprint)
            
        elif mode == 'copilot' and blueprint:
            self._validate_blueprint(blueprint)
            print("[NullHunter] Copilot Mode: Validated user rules + AI scanning remaining features.")
            ai_blueprint = self.scan(data=chunks_iterable)
            merged_blueprint = {**ai_blueprint, **blueprint} 
            clean_data = self.purify(data=chunks_iterable, action_plan=merged_blueprint)
            
        else: # Default: 'auto'
            print("[NullHunter] Auto-Pilot Mode: AI taking full control.")
            ai_blueprint = self.scan(data=chunks_iterable)
            clean_data = self.purify(data=chunks_iterable, action_plan=ai_blueprint)
            
        if return_report:
            print("[NullHunter] Generating Final Verification Report...")
            
        return clean_data