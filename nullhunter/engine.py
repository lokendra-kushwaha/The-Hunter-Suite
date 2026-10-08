import os
import gc
import logging
import psutil
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Union

# ==========================================
# LOGGER CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] NULLHUNTER_ENGINE: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# CUSTOM EXCEPTIONS
# ==========================================
class NullHunterEngineError(Exception):
    """Raised when the execution engine encounters a fatal routing or I/O error."""
    pass

class NullHunterNotImplementedError(Exception):
    """Raised when the blueprint requests a layer that does not exist in the registry."""
    pass

class NullHunterStateError(Exception):
    """Raised when the Master Ledger loses synchronization with the dataframe shape."""
    pass

# ==========================================
# SIMULATED LAYER REGISTRY (For Dependency Injection)
# ==========================================
# In a real framework, this would be imported from nullhunter.layers
LAYER_REGISTRY: Dict[str, Any] = {
    # 'knn_imputer': knn_imputer_function,
    # 'type_caster': type_caster_function,
    # 'outlier_capper': outlier_capper_function,
}

# ==========================================
# STATIC WORKER FUNCTION (Isolated Process Router)
# ==========================================
def _worker_process_chunk(
    chunk: pd.DataFrame, 
    blueprint: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    The Pure DAG Router (Isolated CPU Worker).
    
    This function operates in a completely decoupled manner. It does not perform any
    mathematical operations or memory casting itself. It strictly acts as a router,
    passing the data column to the appropriate standalone layer from the registry based
    on the Scanner's blueprint.
    
    Args:
        chunk (pd.DataFrame): The flattened 2D DataFrame chunk.
        blueprint (Dict): The JSON-like execution instructions from the Scanner.
        
    Returns:
        pd.DataFrame: The processed and optimized chunk.
        
    Raises:
        NullHunterNotImplementedError: If a requested layer is missing.
    """
    processed_chunk = chunk.copy(deep=False)
    
    for col, instructions in blueprint.items():
        if col not in processed_chunk.columns:
            logger.debug(f"Column '{col}' not found in current chunk. Skipping.")
            continue
            
        pipeline = instructions.get("cleaning_pipeline", [])
        mem_opt = instructions.get("memory_optimization", None)
        
        # 1. Smart Lazy Evaluation (Skip heavy math if data is already clean)
        has_nulls = processed_chunk[col].isnull().any()
        
        # Execute Cleaning Pipeline
        for layer_name in pipeline:
            if layer_name == "drop_column":
                processed_chunk = processed_chunk.drop(columns=[col])
                break # Terminate pipeline for this dropped column
                
            # Lazy skip for imputation layers
            if "imputer" in layer_name and not has_nulls:
                continue
                
            # Strict Separation of Concerns: Delegate to External Layer
            if layer_name in LAYER_REGISTRY:
                try:
                    processed_chunk = LAYER_REGISTRY[layer_name](processed_chunk, col)
                except Exception as e:
                    logger.error(f"Layer '{layer_name}' crashed on column '{col}': {str(e)}")
                    raise
            else:
                # The framework enforces development completeness
                raise NullHunterNotImplementedError(
                    f"Scanner requested '{layer_name}' for column '{col}', "
                    f"but this layer is not registered in the framework."
                )
                
        # 2. Execute Memory Optimization (Delegated to type_caster layer)
        if mem_opt and col in processed_chunk.columns:
            if "type_caster" in LAYER_REGISTRY:
                # The engine no longer hardcodes .astype(). It delegates the command.
                processed_chunk = LAYER_REGISTRY["type_caster"](processed_chunk, col, mem_opt)
            else:
                raise NullHunterNotImplementedError(
                    f"Memory optimization '{mem_opt}' requested, but the 'type_caster' "
                    f"layer is missing from the registry."
                )
                
    return processed_chunk


# ==========================================
# THE MASTER ORCHESTRATOR
# ==========================================
class ExecutionEngine:
    """
    The CPU Master Commander for Project NullHunter.
    
    Solely responsible for orchestrating the flow of perfectly-sized chunks (provided 
    by the smart DataLoader) across the available CPU cores. It manages state synchronization 
    (Master Ledger), process pooling, and Disk I/O without interfering with RAM logic.
    """

    def __init__(
        self, 
        loader_ref: Any, 
        flattener_ref: Any, 
        optimizer_ref: Any,
        scanner_ref: Any,
        reconstructor_ref: Any,
        max_cores: Optional[int] = None
    ) -> None:
        """
        Initializes the CPU Commander and internal sub-system references.
        """
        self.loader = loader_ref
        self.flattener = flattener_ref
        self.optimizer = optimizer_ref
        self.scanner = scanner_ref
        self.reconstructor = reconstructor_ref
        
        # Pure CPU Configuration
        self.system_cores = os.cpu_count() or 4
        # Reserve 1 core for the OS/Background tasks to prevent system UI freeze
        self.max_cores = max_cores if max_cores else max(1, self.system_cores - 1)
        
        # State Management
        self.master_ledger: Dict[str, Any] = {}
        self.blueprint: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ExecutionEngine initialized. Commanding {self.max_cores} dedicated CPU cores.")

    def _sync_master_ledger(self, flat_map: Dict[str, Any], opt_map: Dict[str, str]) -> None:
        """
        Fuses structural and semantic maps to maintain the Chain of Custody.
        Ensures the Reconstructor has the exact mathematical reverse-mapping required.
        """
        if not flat_map or not opt_map:
            raise NullHunterStateError("Received empty state maps during Ledger Synchronization.")
            
        for new_name, flat_name in opt_map.items():
            original_structure = flat_map.get(flat_name, flat_name)
            self.master_ledger[new_name] = original_structure
            
        logger.debug("Master Ledger successfully synchronized for downstream reconstruction.")

    def _process_in_memory_fast_path(self, df: pd.DataFrame, mode: str, user_blueprint: dict) -> pd.DataFrame:
        """
        The Zero-Overhead Bypass Mode.
        Activated when the Loader passes a single, monolithic DataFrame (File is tiny).
        """
        logger.info("Engaging Fast Path. Executing sequentially on Main Thread.")
        
        # 1. Pipeline Prep
        flat_df, flat_map = self.flattener.flatten(df)
        opt_df, opt_map = self.optimizer.optimize(flat_df)
        self._sync_master_ledger(flat_map, opt_map)
        
        # 2. Intelligence Routing
        if mode == 'auto':
            self.blueprint = self.scanner.scan(opt_df)
        else:
            self.blueprint = user_blueprint
            
        # 3. Direct Execution
        clean_df = _worker_process_chunk(opt_df, self.blueprint)
        
        # 4. Final Reconstruction
        return self.reconstructor.rebuild(clean_df, self.master_ledger)

    def _process_out_of_core(self, filepath: str, mode: str, user_blueprint: dict) -> str:
        """
        The Core Distribution Protocol (Beast Mode).
        
        Trusts the Loader to provide perfectly sized chunks. Pulls exactly `max_cores` 
        number of chunks, maps them to the CPU pool, and flushes to disk upon completion 
        to maintain a continuous, memory-safe processing stream.
        """
        logger.info(f"Engaging Out-of-Core Processing. Target: {self.max_cores} parallel streams.")
        
        output_file = Path("nullhunter_output.csv")
        is_first_chunk = True
        
        chunk_generator = self.loader.get_chunks()
        
        while True:
            # 1. CPU-Bound Polling (The Engine only requests what the CPU can handle simultaneously)
            batch_chunks = []
            try:
                for _ in range(self.max_cores):
                    batch_chunks.append(next(chunk_generator))
            except StopIteration:
                pass 
                
            if not batch_chunks:
                break # Stream exhausted, exit loop

            # 2. Initialization Phase (Global State Setup)
            if is_first_chunk:
                first_chunk = batch_chunks[0]
                flat_df, flat_map = self.flattener.flatten(first_chunk)
                opt_df, opt_map = self.optimizer.optimize(flat_df)
                self._sync_master_ledger(flat_map, opt_map)
                
                if mode == 'auto':
                    logger.info("Extracting AI Blueprint from leading chunk matrix...")
                    self.blueprint = self.scanner.scan(opt_df)
                else:
                    self.blueprint = user_blueprint
                
                is_first_chunk = False

            # 3. Pre-Processing Dispatch Queue
            prepared_batch = []
            for c in batch_chunks:
                f_c, _ = self.flattener.flatten(c)
                o_c, _ = self.optimizer.optimize(f_c)
                prepared_batch.append(o_c)

            processed_batch = []
            logger.info(f"Deploying batch of {len(prepared_batch)} chunks across active CPUs...")
            
            # 4. Parallel Core Execution
            with ProcessPoolExecutor(max_workers=self.max_cores) as executor:
                # futures list preserves the state; as_completed yields them as they finish
                futures = [
                    executor.submit(_worker_process_chunk, chunk, self.blueprint) 
                    for chunk in prepared_batch
                ]
                for future in as_completed(futures):
                    # Will implicitly raise any NullHunterNotImplementedError that occurred in the worker
                    processed_batch.append(future.result())

            # 5. Continuous Disk Flush (OOM Prevention)
            for clean_chunk in processed_batch:
                final_chunk = self.reconstructor.rebuild(clean_chunk, self.master_ledger)
                
                write_mode = 'w' if not output_file.exists() else 'a'
                write_header = not output_file.exists()
                final_chunk.to_csv(output_file, mode=write_mode, index=False, header=write_header)
            
            # 6. Strict Memory Sweeping
            del batch_chunks
            del prepared_batch
            del processed_batch
            gc.collect()

        logger.info(f"Processing sequence complete. Data serialized to: {output_file}")
        return str(output_file)

    def execute(
        self, 
        data: Union[str, pd.DataFrame, pd.Series], 
        mode: str = 'auto', 
        blueprint: Optional[Dict] = None
    ) -> Union[pd.DataFrame, pd.Series, str]:
        """
        The External Entry Point.
        Normalizes single-dimension Series and routes based on input location (RAM vs Disk).
        """
        was_series = False
        
        if isinstance(data, pd.Series):
            was_series = True
            data = data.to_frame()
            logger.info("Series dimensional expansion applied. Escalating to DataFrame.")

        if isinstance(data, pd.DataFrame):
            result = self._process_in_memory_fast_path(data, mode, blueprint)
            return result.squeeze() if was_series else result
            
        elif isinstance(data, str) or isinstance(data, Path):
            # Engine delegates RAM evaluation ENTIRELY to the Loader.
            # If the Loader set fast_path=True during its internal math, it will 
            # yield exactly 1 large DataFrame when get_chunks() is called.
            # However, for API consistency, we can check the Loader's determined state here:
            
            if self.loader._fast_path:
                logger.info("Delegating to Main Thread (Loader specified Fast Path).")
                # We pull the single monolith chunk the loader prepared
                df = next(self.loader.get_chunks())
                return self._process_in_memory_fast_path(df, mode, blueprint)
            else:
                logger.info("Delegating to Process Pool (Loader specified Chunk Path).")
                return self._process_out_of_core(str(data), mode, blueprint)
                
        else:
            raise NullHunterEngineError(f"Engine cannot process datatype: {type(data)}")