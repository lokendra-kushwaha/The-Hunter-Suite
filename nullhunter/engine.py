"""
NullHunter Execution Engine (The Pipeline Orchestrator)
===================================================================

This module acts as the isolated Pass-2 Execution Pipeline for the NullHunter framework.
It operates purely on the "Delegation Pattern", receiving planning assets (Blueprint, 
Master Ledger) from the API Gateway and orchestrating the massive data flow utilizing 
UniversalParallelRunner.

Key Architectural Pillars:
1. Dynamic RAM Buffering (Smart Concatenation): Eliminates Disk I/O bottlenecks by 
   accumulating reconstructed chunks in RAM up to a safe limit before monolithic writes.
2. Multi-Format Atomic I/O: Supports CSV, Parquet, and JSON chunked writing natively.
3. Pre-Execution Blueprint Validation: Statically verifies tool existence before pipeline start.
4. Adaptive Memory Shield: Monitors OS memory limits and forces aggressive garbage collection.
5. Dynamic Dispatch (Registry Pattern): Eliminates conditional spaghetti logic via UDFs.
"""

import os
import gc
import time
import signal
import logging
import itertools
import traceback
import psutil
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
# TELEMETRY DATA STRUCTURES
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
    traceback_log: Optional[str] = None


# ==========================================
# TOOL REGISTRY (The Algorithm Box)
# ==========================================
class _ToolRegistry:
    """
    A robust registry for maintaining analytical tools.
    Provides isolation, validation, and dynamic dispatching capabilities.
    """
    def __init__(self) -> None:
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
    AFTER all mathematical operations to prevent precision loss or TypeErrors.
    
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
        
        # Smart Lazy Evaluation
        has_nulls = processed_chunk[col].isnull().any()
        
        # 1. Execute Mathematical Pipeline (Dynamic Dispatch)
        for tool_name in pipeline:
            # Skip imputation algorithms if the column has zero nulls in this specific chunk
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
                metrics.traceback_log = traceback.format_exc()
                raise RuntimeError(metrics.error_msg)
                
            # If a tool (like drop_column) removed the column, break the inner loop
            if col not in processed_chunk.columns:
                break
                
        # 2. Execute Memory Optimization (Strictly separate and executed last)
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
# THE EXECUTION ENGINE (Orchestrator)
# ==========================================
class ExecutionEngine:
    """
    The Execution Pipeline Orchestrator.
    
    Operates strictly in Pass-2. Takes the pre-calculated Blueprint and Master Ledger 
    from the Gateway and orchestrates the massive data flow utilizing Smart Write Buffering.
    
    Responsibilities:
    - Maintains OS stability by reserving CPU cores.
    - Manages Dynamic RAM Buffering to eliminate Disk I/O bottlenecks.
    - Pre-validates AI blueprints against the Tool Registry.
    - Handles exponential backoff for multi-format disk writes (CSV/Parquet).
    """
    
    def __init__(
        self, 
        loader_ref: Any, 
        reconstructor_ref: Any,
        parallel_runner_ref: UniversalParallelRunner,
        write_buffer_limit_mb: int = 50,
        max_cores: Optional[int] = None,
        io_retries: int = 4,
        io_backoff_factor: float = 1.5,
        memory_shield_threshold: float = 90.0
    ) -> None:
        """
        Initializes the Engine with external dependencies and robust architectural limits.
        
        Args:
            loader_ref (Any): Reference to the instantiated DataLoader.
            reconstructor_ref (Any): Reference to the instantiated Reconstructor.
            parallel_runner_ref (UniversalParallelRunner): The God-Mode executor.
            write_buffer_limit_mb (int): Max RAM allocated for chunk concatenation.
            max_cores (Optional[int]): Hard limit for cores. Defaults to OS Cores - 1.
            io_retries (int): Maximum retry attempts for disk writes.
            io_backoff_factor (float): Multiplier for exponential sleep during disk lock.
            memory_shield_threshold (float): % of system RAM usage triggering aggressive GC.
        """
        self.loader = loader_ref
        self.reconstructor = reconstructor_ref
        self.runner = parallel_runner_ref
        
        # Enforce minimum buffer size to prevent useless micro-writes
        self.write_buffer_limit_mb = max(10, write_buffer_limit_mb)
        
        self.system_cores = os.cpu_count() or 4
        self.max_cores = max_cores if max_cores else max(1, self.system_cores - 1)
        
        self.io_retries = max(1, io_retries)
        self.io_backoff_factor = max(1.1, io_backoff_factor)
        self.memory_shield_threshold = memory_shield_threshold
        
        self._shutdown_flag = False
        self._write_buffer: List[pd.DataFrame] = []
        self._current_buffer_weight_mb: float = 0.0
        self._is_first_write = True
        
        self._register_signals()
        
        logger.info(
            f"ExecutionEngine Armed | Core Limit: {self.max_cores} | "
            f"Buffer Limit: {self.write_buffer_limit_mb}MB | I/O Retries: {self.io_retries}"
        )

    def _register_signals(self) -> None:
        """Registers OS signals to allow graceful shutdown without corrupting the Output file."""
        try:
            signal.signal(signal.SIGINT, self._handle_shutdown)
            signal.signal(signal.SIGTERM, self._handle_shutdown)
        except ValueError:
            # Safe to ignore if running in a child thread
            pass

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        """Gracefully halts the pipeline on Ctrl+C."""
        logger.warning("\n[ALERT] Termination signal received. Halting pipeline gracefully...")
        self._shutdown_flag = True

    def _verify_blueprint_integrity(self, blueprint: Dict[str, Dict[str, Any]]) -> None:
        """
        Pre-Execution Static Validation.
        Scans the entire blueprint to ensure all requested tools exist in the registry.
        Prevents the pipeline from crashing mid-execution after hours of processing.
        """
        logger.debug("Running Static Validation on AI Blueprint...")
        missing_tools = set()
        
        for col, instructions in blueprint.items():
            pipeline = instructions.get("cleaning_pipeline", [])
            for tool in pipeline:
                if not registry.get_tool(tool):
                    missing_tools.add(tool)
                    
        if missing_tools:
            raise NullHunterNotImplementedError(
                f"Blueprint Integrity Check Failed! The following AI tools were requested "
                f"by the Scanner but are missing from the Engine Registry: {list(missing_tools)}"
            )
        logger.info("Blueprint Integrity Verified. All required tools are active.")


    def _safe_disk_io_writer(self, df: pd.DataFrame, path: Path, is_first: bool) -> None:
        """
        The Abstracted Multi-Format Atomic Writer.
        Writes DataFrames to disk with an Exponential Backoff Retry mechanism.
        Automatically detects file extensions (.csv, .parquet, .json).
        
        Args:
            df (pd.DataFrame): The final chunk or concatenated buffer to write.
            path (Path): Destination file path.
            is_first (bool): Determines if headers should be written and file overwritten.
            
        Raises:
            NullHunterDiskIOError: If all retries fail.
        """
        file_ext = path.suffix.lower()
        write_mode = 'w' if is_first else 'a'
        write_header = is_first

        for attempt in range(self.io_retries):
            try:
                if file_ext == '.parquet':
                    # Parquet requires engine-specific appending logic
                    if is_first:
                        df.to_parquet(path, engine='pyarrow', index=False)
                    else:
                        import pyarrow as pa
                        import pyarrow.parquet as pq
                        table = pa.Table.from_pandas(df, preserve_index=False)
                        with pq.ParquetWriter(path, table.schema) as writer:
                            writer.write_table(table)
                elif file_ext == '.json':
                    df.to_json(path, orient='records', lines=True, mode=write_mode)
                else:
                    # Default CSV writer
                    df.to_csv(path, mode=write_mode, index=False, header=write_header)
                    
                return  # Atomic Write Successful
                
            except (PermissionError, OSError) as e:
                logger.warning(f"Disk locked/busy. Retry {attempt + 1}/{self.io_retries} due to: {str(e)}")
                time.sleep(self.io_backoff_factor ** attempt)
            except Exception as e:
                raise NullHunterDiskIOError(f"Critical I/O serialization failure: {str(e)}")
                
        raise NullHunterDiskIOError(f"Failed to flush RAM to '{path}' after {self.io_retries} attempts.")


    def _flush_buffer_to_disk(self, output_file: Path) -> None:
        """
        Concatenates all reconstructed chunks residing in the RAM Buffer and 
        flushes them to disk using the generalized safe I/O writer.
        """
        if not self._write_buffer:
            return
            
        logger.info(f"Flushing {self._current_buffer_weight_mb:.1f}MB Concatenated RAM Buffer to disk...")
        
        try:
            # Smart RAM Concatenation (High Speed, Single Allocation)
            merged_df = pd.concat(self._write_buffer, ignore_index=True)
        except Exception as e:
            raise NullHunterEngineError(f"RAM Buffer Concatenation failed: {str(e)}")
            
        # Delegate to the standalone Atomic Writer
        self._safe_disk_io_writer(df=merged_df, path=output_file, is_first=self._is_first_write)
        
        # Update State Flags & Free Memory
        self._is_first_write = False
        self._write_buffer.clear()
        self._current_buffer_weight_mb = 0.0


    def execute_pipeline(
        self, 
        destination: Union[str, Path],
        master_ledger: Dict[str, Any],
        blueprint: Dict[str, Dict[str, Any]],
        keep_optimized_headers: bool = False,
        fail_fast: bool = False
    ) -> Dict[str, Any]:
        """
        The Master Execution Loop (Pass 2).
        
        Streams batches from the Loader, processes them via Runner, reconstructs their 
        original shapes, and utilizes the Smart Buffer to eliminate Disk I/O bottlenecks.
        
        Args:
            destination (Union[str, Path]): Where the final cleaned data should be saved.
            master_ledger (Dict[str, Any]): Mapping for the reconstructor.
            blueprint (Dict[str, Dict[str, Any]]): AI execution instructions.
            keep_optimized_headers (bool): Preserve ML-ready names during reconstruction.
            fail_fast (bool): If True, aborts the pipeline upon a single chunk failure.
            
        Returns:
            Dict[str, Any]: A massive, SaaS-level telemetry report.
        """
        start_time = time.perf_counter()
        output_file = Path(destination)
        
        # Validate blueprint BEFORE starting execution
        self._verify_blueprint_integrity(blueprint)
        
        telemetry = {
            "status": "in_progress",
            "total_chunks_processed": 0,
            "total_chunks_failed": 0,
            "total_memory_saved_mb": 0.0,
            "buffer_flushes_executed": 0,
            "failure_logs": [],
            "execution_time_sec": 0.0,
            "destination_path": str(output_file)
        }
        
        logger.info(f"Initiating Pass-2 Execution Pipeline. Output target: {output_file}")
        chunk_generator: Generator = self.loader.get_chunks()
        
        while not self._shutdown_flag:
            # 1. Adaptive Memory Shield
            if psutil.virtual_memory().percent > self.memory_shield_threshold:
                logger.warning(f"Memory Shield Activated (RAM > {self.memory_shield_threshold}%). Forcing GC.")
                gc.collect()

            # 2. Pull Batch strictly limited by available Cores
            batch_chunks = list(itertools.islice(chunk_generator, self.max_cores))
            if not batch_chunks:
                break 
                
            # 3. Parallel Processing via Runner
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

            # 4. Shape Reconstruction & Buffer Routing
            for result_tuple in runner_results:
                if not isinstance(result_tuple, tuple) or len(result_tuple) != 2:
                    continue
                    
                clean_chunk, chunk_metrics = result_tuple
                telemetry["total_chunks_processed"] += 1
                telemetry["total_memory_saved_mb"] += getattr(chunk_metrics, 'memory_saved_mb', 0.0)
                
                # Reconstruct Shape (Layer 3.0)
                if master_ledger:
                    final_chunk = self.reconstructor.rebuild(
                        chunk=clean_chunk, 
                        master_ledger=master_ledger,
                        keep_optimized_headers=keep_optimized_headers,
                        inplace=True 
                    )
                else:
                    final_chunk = clean_chunk
                    
                # Calculate chunk weight and route to Smart Buffer
                chunk_weight = final_chunk.memory_usage(deep=True).sum() / (1024 ** 2)
                self._write_buffer.append(final_chunk)
                self._current_buffer_weight_mb += chunk_weight
                
                # Check Flush Threshold
                if self._current_buffer_weight_mb >= self.write_buffer_limit_mb:
                    self._flush_buffer_to_disk(output_file)
                    telemetry["buffer_flushes_executed"] += 1

            # 5. Aggressive Memory Sweeping (GC)
            del batch_chunks
            del runner_results
            gc.collect()
            
        # 6. Residual Flush (Drain the remaining buffer)
        if self._write_buffer and telemetry["status"] != "aborted":
            self._flush_buffer_to_disk(output_file)
            telemetry["buffer_flushes_executed"] += 1
            
        if self._shutdown_flag:
            telemetry["status"] = "interrupted_by_user"
        elif telemetry["status"] != "aborted":
            telemetry["status"] = "success"
            
        telemetry["execution_time_sec"] = round(time.perf_counter() - start_time, 4)
        telemetry["total_memory_saved_mb"] = round(telemetry["total_memory_saved_mb"], 2)
        
        logger.info(
            f"Pipeline Completed [{telemetry['status']}] in {telemetry['execution_time_sec']}s. "
            f"Processed: {telemetry['total_chunks_processed']} | "
            f"RAM Saved: {telemetry['total_memory_saved_mb']} MB | "
            f"Disk Writes (Flushes): {telemetry['buffer_flushes_executed']}"
        )
                    
        return telemetry  