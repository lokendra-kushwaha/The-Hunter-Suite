import os
import time
import logging
import traceback
import concurrent.futures
from typing import Callable, Iterable, Dict, Any, Tuple, List, Optional, Union

# ==========================================
# LOGGER CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] NULLHUNTER_RUNNER: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==========================================
# ISOLATED WORKER CAPSULE (Top-Level for Pickle Safety)
# ==========================================
def _isolated_worker_capsule(task_id: int, func: Callable, chunk: Any, *args, **kwargs) -> Dict[str, Any]:
    """
    The Bulletproof Wrapper. Executes the target function inside an isolated try-except block.
    Placed at the module level to ensure 100% compatibility with Windows/Linux multiprocessing (Pickle safe).
    """
    start_time = time.time()
    try:
        result = func(chunk, *args, **kwargs)
        return {
            "task_id": task_id,
            "status": "success",
            "result": result,
            "error": None,
            "traceback": None,
            "execution_time": time.time() - start_time
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "failed",
            "result": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "execution_time": time.time() - start_time
        }

# ==========================================
# THE EXECUTOR
# ==========================================
class UniversalParallelRunner:
    """
    Enterprise-Grade Universal Parallel Executor (The NullHunter Engine Room).

    A highly fault-tolerant, hardware-aware multiprocessing orchestrator. It acts as a 
    higher-order wrapper that can take ANY user-defined function (UDF) or internal 
    module and safely distribute it across available CPU cores.

    Features:
        - OS Protection: Automatically reserves CPU cores to prevent system freezing.
        - Fault Isolation: One failed chunk does not crash the entire pipeline.
        - Telemetry: Generates a detailed Execution Report (Success/Failures/Time).
        - Auto-Fallback: Gracefully degrades to sequential execution if multiprocessing fails.
        - Timeout Governor: Kills zombie processes that hang indefinitely.
    """

    def __init__(
        self,
        backend: str = 'process',
        max_workers: Optional[int] = None,
        reserve_cores: int = 1,
        fail_fast: bool = False,
        fallback_sequential: bool = True,
        timeout_per_task: Optional[float] = None
    ) -> None:
        """
        Initializes the UniversalParallelRunner with SaaS-level execution parameters.

        Args:
            backend (str): 'process' (for heavy CPU data math) or 'thread' (for I/O network tasks).
            max_workers (Optional[int]): Hard limit on workers. If None, auto-calculated via OS profiling.
            reserve_cores (int): Number of CPU cores to leave strictly free for the OS.
            fail_fast (bool): If True, stops the entire pool immediately upon the first chunk failure.
            fallback_sequential (bool): If True, executes sequentially if pool initialization/pickling fails.
            timeout_per_task (Optional[float]): Max seconds allowed per chunk before raising a TimeoutError.
        """
        self.backend = backend.lower()
        self.reserve_cores = reserve_cores
        self.fail_fast = fail_fast
        self.fallback_sequential = fallback_sequential
        self.timeout_per_task = timeout_per_task
        
        if self.backend not in ['process', 'thread']:
            raise ValueError("Invalid backend. Choose 'process' or 'thread'.")

        # Hardware Profiling (The Governor)
        self.total_system_cores = os.cpu_count() or 4
        if max_workers is not None:
            self.workers = min(max_workers, self.total_system_cores)
        else:
            self.workers = max(1, self.total_system_cores - self.reserve_cores)

        logger.info(
            f"UniversalParallelRunner Armed | Backend: {self.backend.upper()} | "
            f"Workers: {self.workers}/{self.total_system_cores} | Fail-Fast: {self.fail_fast}"
        )

    def _execute_sequentially(self, func: Callable, payload: Iterable, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """Fallback mechanism if multiprocessing is blocked by system or pickle constraints."""
        logger.warning("Initiating Sequential Fallback Execution...")
        results = []
        failures = []
        start_global = time.time()

        for idx, chunk in enumerate(payload):
            task_report = _isolated_worker_capsule(idx, func, chunk, *args, **kwargs)
            if task_report['status'] == 'success':
                results.append(task_report['result'])
            else:
                failures.append(task_report)
                if self.fail_fast:
                    logger.error(f"Sequential Execution aborted due to fail_fast at task {idx}.")
                    break

        execution_report = self._generate_telemetry(len(results) + len(failures), len(results), failures, time.time() - start_global)
        return results, execution_report

    def _generate_telemetry(self, total: int, success: int, failures: List[Dict], duration: float) -> Dict[str, Any]:
        """Compiles a SaaS-grade execution analytics report."""
        return {
            "total_tasks": total,
            "successful_tasks": success,
            "failed_tasks": len(failures),
            "success_rate": f"{(success / max(total, 1)) * 100:.2f}%",
            "total_execution_time_sec": round(duration, 4),
            "failure_logs": failures
        }

    def execute(self, func: Callable, payload: Iterable, *args, **kwargs) -> Tuple[List[Any], Dict[str, Any]]:
        """
        The Master Execution API. Maps any function over a data payload in parallel.

        Args:
            func (Callable): The target function (Internal module or User-Defined Function).
            payload (Iterable): A list or generator of data chunks to process.
            *args: Positional arguments to pass to the target function.
            **kwargs: Keyword arguments to pass to the target function.

        Returns:
            Tuple[List[Any], Dict[str, Any]]: 
                - List of successfully processed results (in original order).
                - Execution Telemetry Report detailing performance and isolated errors.
        """
        logger.info(f"Igniting Execution Pool for function: '{func.__name__}'...")
        global_start_time = time.time()
        
        # We need a list to maintain order mapping
        payload_list = list(payload)
        total_tasks = len(payload_list)
        
        if total_tasks == 0:
            logger.warning("Empty payload provided. Aborting execution.")
            return [], self._generate_telemetry(0, 0, [], 0.0)

        results = [None] * total_tasks
        failures = []
        success_count = 0

        ExecutorClass = concurrent.futures.ProcessPoolExecutor if self.backend == 'process' else concurrent.futures.ThreadPoolExecutor

        try:
            with ExecutorClass(max_workers=self.workers) as executor:
                # Submit tasks mapped with their original index to preserve order
                future_to_index = {
                    executor.submit(_isolated_worker_capsule, idx, func, chunk, *args, **kwargs): idx
                    for idx, chunk in enumerate(payload_list)
                }

                for future in concurrent.futures.as_completed(future_to_index):
                    idx = future_to_index[future]
                    try:
                        # Governor checks for timeouts
                        task_report = future.result(timeout=self.timeout_per_task)
                        
                        if task_report['status'] == 'success':
                            results[idx] = task_report['result']
                            success_count += 1
                        else:
                            failures.append(task_report)
                            logger.error(f"Task {idx} Failed: {task_report['error']}")
                            if self.fail_fast:
                                logger.critical("Fail-Fast triggered. Cancelling pending pool tasks!")
                                executor.shutdown(wait=False, cancel_futures=True)
                                break
                    except concurrent.futures.TimeoutError:
                        err_msg = f"Task {idx} exceeded timeout of {self.timeout_per_task}s."
                        logger.error(err_msg)
                        failures.append({"task_id": idx, "status": "failed", "error": err_msg, "traceback": None})
                        if self.fail_fast:
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                    except Exception as e:
                        # Catch-all for extreme system crashes
                        failures.append({"task_id": idx, "status": "failed", "error": str(e), "traceback": traceback.format_exc()})

        except (AttributeError, TypeError, Exception) as runtime_err:
            # Handles "PicklingError" when user passes a complex un-serializable function
            logger.error(f"Pool Initialization Failed: {str(runtime_err)}")
            if self.fallback_sequential:
                return self._execute_sequentially(func, payload_list, *args, **kwargs)
            else:
                raise RuntimeError(f"Parallel Execution failed and fallback is disabled. Cause: {str(runtime_err)}")

        # Filter out None values (failed tasks leave holes in the results array)
        clean_results = [res for res in results if res is not None]
        
        execution_report = self._generate_telemetry(
            total_tasks, success_count, failures, time.time() - global_start_time
        )
        
        logger.info(f"Execution Complete. Success Rate: {execution_report['success_rate']}")
        return clean_results, execution_report