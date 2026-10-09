"""
NullHunter Execution Engine
=================================================

This module acts as the isolated Pass-2 Execution Pipeline for the NullHunter framework.
It operates purely on the "Delegation Pattern", receiving planning assets (Blueprint, 
Master Ledger) from the core.py and executing them using the 
UniversalParallelRunner.

Key Architectural Pillars:
1. Dynamic Dispatch (Registry Pattern): Eliminates if-else spaghetti logic.
2. Dumb Core Routing: Defers RAM management entirely to the Loader. It only maps 
   batches strictly to the available CPU count to prevent OS UI freezing.
3. Fault Tolerance & Atomic I/O: Implements exponential backoff for disk writes.
4. Granular Telemetry: Tracks microsecond-level execution times per transformation.
"""

import os
import gc
import time
import signal
import logging
import itertools
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union, Callable, Tuple, Generator

from nullhunter.utils.runner import UniversalParallelRunner 


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
    """Raised when the execution engine encounters a fatal orchestration error."""
    pass

class NullHunterNotImplementedError(Exception):
    """Raised when the blueprint requests a tool that does not exist in the registry."""
    pass

class NullHunterDiskIOError(Exception):
    """Raised when the engine fails to write to the disk after maximum retries."""
    pass


# ==========================================
# DATA STRUCTURES
# ==========================================
@dataclass
class ChunkTelemetry:
    """
    Data class for storing micro-metrics of a single chunk's execution.
    Returned alongside the dataframe by the UDF to feed the master telemetry.
    """
    chunk_id: str
    total_time_sec: float = 0.0
    tools_applied: List[str] = field(default_factory=list)
    memory_saved_mb: float = 0.0
    error_flag: bool = False
    error_msg: str = ""


# ==========================================
# TOOL REGISTRY
# ==========================================
class _ToolRegistry:
    """
    A robust registry for maintaining analytical tools.
    Provides isolation, validation, and dynamic dispatching capabilities.
    """
    def __init__(self):
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        """
        Decorator to dynamically register a processing tool into the engine's armory.
        
        Args:
            name (str): The exact string key present in the AI Blueprint.
            
        Returns:
            Callable: The decorator function.
        """
        def decorator(func: Callable) -> Callable:
            if not callable(func):
                raise TypeError(f"Registered tool '{name}' must be a callable function.")
            if name in self._tools:
                logger.warning(f"Overwriting existing tool '{name}' in the registry.")
            self._tools[name] = func
            return func
        return decorator

    def get_tool(self, name: str) -> Optional[Callable]:
        """Retrieves a tool by name securely."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """Returns a list of all currently registered tools."""
        return list(self._tools.keys())

# Instantiate the global Registry
registry = _ToolRegistry()


# --- Dummy Implementations for Registry (To be moved to nullhunter.tools later) ---
@registry.register("drop_column")
def _tool_drop_column(chunk: pd.DataFrame, col: str, **kwargs) -> pd.DataFrame:
    if col in chunk.columns:
        return chunk.drop(columns=[col])
    return chunk


# ==========================================
# THE ISOLATED CPU WORKER (UDF for Runner)
# ==========================================
def _apply_blueprint_to_chunk(
    chunk: pd.DataFrame, 
    blueprint: Dict[str, Dict[str, Any]]
) -> Tuple[pd.DataFrame, ChunkTelemetry]:
    """
    The Pure DAG Router (Worker UDF).
    
    Operates in isolated global scope for Pickle serialization. It applies transformations
    based on the AI blueprint and records granular execution metrics.
    
    Note on Execution Order: Memory Optimization (type casting) is strictly executed 
    AFTER all mathematical operations (imputation/capping) to prevent precision loss 
    or TypeErrors during execution.
    
    Args:
        chunk (pd.DataFrame): The flattened 2D DataFrame chunk.
        blueprint (Dict[str, Dict[str, Any]]): Execution instructions per column.
        
    Returns:
        Tuple[pd.DataFrame, ChunkTelemetry]: The cleaned chunk and its micro-metrics.
        
    Raises:
        RuntimeError: If a tool crashes during execution.
        NullHunterNotImplementedError: If a requested tool is missing.
    """
    start_time = time.perf_counter()
    initial_mem = chunk.memory_usage(deep=True).sum() / (1024 ** 2)
    
    # Defensive copy to prevent modifying the original reference
    processed_chunk = chunk.copy(deep=False)
    
    # Use index 0 as a mock chunk ID if real ID isn't present
    chunk_id = str(processed_chunk.index[0]) if not processed_chunk.empty else "empty_chunk"
    metrics = ChunkTelemetry(chunk_id=chunk_id)
    
    for col, instructions in blueprint.items():
        if col not in processed_chunk.columns:
            continue
            
        pipeline = instructions.get("cleaning_pipeline", [])
        mem_opt = instructions.get("memory_optimization", None)
        
        # 1. Smart Lazy Evaluation
        has_nulls = processed_chunk[col].isnull().any()
        
        # 2. Execute Mathematical Pipeline (Dynamic Dispatch)
        for tool_name in pipeline:
            if "imputer" in tool_name and not has_nulls:
                continue
                
            active_tool = registry.get_tool(tool_name)
            if not active_tool:
                raise NullHunterNotImplementedError(
                    f"Blueprint requested '{tool_name}' for '{col}', but it is missing."
                )
                
            try:
                processed_chunk = active_tool(processed_chunk, col)
                metrics.tools_applied.append(f"{col}:{tool_name}")
            except Exception as e:
                metrics.error_flag = True
                metrics.error_msg = f"Tool '{tool_name}' failed on '{col}': {str(e)}"
                raise RuntimeError(metrics.error_msg)
                
            if col not in processed_chunk.columns:
                break
                
        # 3. Execute Memory Optimization (Strictly separate and executed last)
        if mem_opt and col in processed_chunk.columns:
            caster = registry.get_tool("type_caster")
            if caster:
                try:
                    processed_chunk = caster(processed_chunk, col, target_type=mem_opt)
                    metrics.tools_applied.append(f"{col}:type_caster")
                except Exception as e:
                    logger.debug(f"Memory optimization failed on '{col}': {e}")
            else:
                logger.debug("Skipping memory optimization: 'type_caster' not registered.")
                
    # Finalize Metrics
    final_mem = processed_chunk.memory_usage(deep=True).sum() / (1024 ** 2)
    metrics.memory_saved_mb = max(0.0, initial_mem - final_mem)
    metrics.total_time_sec = time.perf_counter() - start_time
    
    return processed_chunk, metrics


# ==========================================
# THE Execution Engine
# ==========================================
class ExecutionEngine:
    """
    The Execution Pipeline Worker.
    
    Operates strictly in Pass-2. Takes the pre-calculated Blueprint and Master Ledger 
    from the core.py and orchestrates the massive data flow.
    
    Responsibilities:
    - Maintains OS stability by reserving exactly 1 CPU core.
    - Streams data efficiently using `itertools.islice`.
    - Handles exponential backoff for disk I/O to prevent file locking crashes.
    """
    
    def __init__(
        self, 
        loader_ref: Any, 
        reconstructor_ref: Any,
        parallel_runner_ref: UniversalParallelRunner,
        max_cores: Optional[int] = None
    ) -> None:
        """
        Initializes the Engine with external dependencies.
        
        Args:
            loader_ref (Any): Reference to the instantiated DataLoader.
            reconstructor_ref (Any): Reference to the instantiated Reconstructor.
            parallel_runner_ref (UniversalParallelRunner): The God-Mode executor.
            max_cores (Optional[int]): Hard limit for cores. Defaults to N-1.
        """
        self.loader = loader_ref
        self.reconstructor = reconstructor_ref
        self.runner = parallel_runner_ref
        
        self.system_cores = os.cpu_count() or 4
        # N-1 Logic: Prevents OS freezing by keeping 1 core free.
        self.max_cores = max_cores if max_cores else max(1, self.system_cores - 1)
        
        self._shutdown_flag = False
        self._register_signals()
        
        logger.info(f"ExecutionEngine Armed. Target Capacity: {self.max_cores} parallel streams.")


    def _register_signals(self) -> None:
        """Registers OS signals to allow graceful shutdown without corrupting the CSV."""
        try:
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except ValueError:
            # Occurs if not running in the main thread; safe to ignore.
            pass

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Gracefully halts the pipeline on Ctrl+C."""
        logger.warning("\n[ALERT] Termination signal received. Halting pipeline gracefully...")
        self._shutdown_flag = True


    def _safe_disk_write(self, df: pd.DataFrame, path: Path, is_first: bool) -> None:
        """
        Writes dataframe to disk with an Exponential Backoff Retry mechanism.
        Protects against temporary OS-level file locks.
        
        Args:
            df (pd.DataFrame): The final reconstructed chunk.
            path (Path): Destination file path.
            is_first (bool): Determines if headers should be written and file overwritten.
            
        Raises:
            NullHunterDiskIOError: If all retries fail.
        """
        max_retries = 3
        write_mode = 'w' if is_first else 'a'
        write_header = is_first
        
        for attempt in range(max_retries):
            try:
                df.to_csv(path, mode=write_mode, index=False, header=write_header)
                return  # Success
            except PermissionError as e:
                logger.warning(f"Disk locked by OS. Retry {attempt + 1}/{max_retries}...")
                time.sleep(1.5 ** attempt) # Exponential backoff: 1s, 1.5s, 2.25s
            except Exception as e:
                raise NullHunterDiskIOError(f"Unexpected I/O failure: {str(e)}")
                
        raise NullHunterDiskIOError(f"Failed to write to '{path}' after {max_retries} attempts.")


    def execute_pipeline(
        self, 
        destination: Union[str, Path],
        master_ledger: Dict[str, Any],
        blueprint: Dict[str, Dict[str, Any]],
        fail_fast: bool = False
    ) -> Dict[str, Any]:
        """
        The Master Execution Loop.
        
        Streams batches sequentially from the Loader matching the core capacity, 
        processes them in parallel, and flushes to disk.
        
        Args:
            destination (Union[str, Path]): Where the final cleaned CSV should be saved.
            master_ledger (Dict[str, Any]): Mapping for the reconstructor. Empty if bypassed.
            blueprint (Dict[str, Dict[str, Any]]): Execution instructions for the tools.
            fail_fast (bool): If True, aborts the pipeline upon a single chunk failure.
            
        Returns:
            Dict[str, Any]: A massive, SaaS-level telemetry report.
        """
        start_time = time.perf_counter()
        output_file = Path(destination)
        
        telemetry = {
            "status": "in_progress",
            "total_chunks_processed": 0,
            "total_chunks_failed": 0,
            "total_memory_saved_mb": 0.0,
            "failure_logs": [],
            "execution_time_sec": 0.0,
            "destination_path": str(output_file)
        }
        
        logger.info(f"Initiating Pass-2 Execution Pipeline. Output: {output_file}")
        
        # Engine is "dumb" to RAM. It only pulls what it can compute simultaneously.
        chunk_generator: Generator = self.loader.get_chunks()
        is_first_write = True
        
        while not self._shutdown_flag:
            # 1. Pull exact batch size using itertools (Clean & Pythonic)
            batch_chunks = list(itertools.islice(chunk_generator, self.max_cores))
            
            if not batch_chunks:
                break # Stream completely exhausted
                
            logger.info(f"Dispatching batch of {len(batch_chunks)} chunks to Runner...")
            
            # 2. Execute via God-Mode Runner
            # The Runner returns a list of tuples: [(df, metrics), (df, metrics), ...]
            runner_results, run_report = self.runner.execute(
                func=_apply_blueprint_to_chunk,
                payload=batch_chunks,
                blueprint=blueprint
            )
            
            telemetry["total_chunks_failed"] += run_report.get("failed_tasks", 0)
            if run_report.get("failed_tasks", 0) > 0:
                telemetry["failure_logs"].extend(run_report.get("failure_logs", []))
                if fail_fast:
                    logger.error("Fail-fast triggered. Aborting pipeline.")
                    telemetry["status"] = "aborted"
                    break

            # 3. Process Success Returns & Safe Disk Flush
            for result_tuple in runner_results:
                if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                    continue
                    
                clean_chunk, chunk_metrics = result_tuple
                
                # Aggregate micro-metrics
                telemetry["total_chunks_processed"] += 1
                telemetry["total_memory_saved_mb"] += getattr(chunk_metrics, 'memory_saved_mb', 0.0)
                
                # O(1) Reconstruction
                if master_ledger:
                    final_chunk = self.reconstructor.rebuild(clean_chunk, master_ledger)
                else:
                    final_chunk = clean_chunk
                    
                # Atomic Write
                self._safe_disk_write(final_chunk, output_file, is_first_write)
                is_first_write = False
            
            # 4. Aggressive Memory Sweeping (GC)
            del batch_chunks
            del runner_results
            gc.collect()
            
        if self._shutdown_flag:
            telemetry["status"] = "interrupted_by_user"
        elif telemetry["status"] != "aborted":
            telemetry["status"] = "success"
            
        telemetry["execution_time_sec"] = round(time.perf_counter() - start_time, 4)
        telemetry["total_memory_saved_mb"] = round(telemetry["total_memory_saved_mb"], 2)
        
        logger.info(
            f"Pipeline Completed [{telemetry['status']}] in {telemetry['execution_time_sec']}s. "
            f"Processed: {telemetry['total_chunks_processed']} | "
            f"Failed: {telemetry['total_chunks_failed']} | "
            f"RAM Saved: {telemetry['total_memory_saved_mb']} MB"
        )
                    
        return telemetry
    