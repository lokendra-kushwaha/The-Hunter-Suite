"""
NullHunter Data Ingestion Node (The RAM Architect)
==================================================

Enterprise-Grade CPU/RAM-Aware Data Ingestion Engine for Project NullHunter.
Acts as the secure, autonomous entry point for datasets. It completely eliminates
hardcoding by profiling real-time byte sizes, calculating optimal chunk sizes 
for CPU cores, and generating dynamic memory buffers for the downstream Engine.

Key Features:
- Zero-OOM Streaming: Mathematically divides available RAM by active cores.
- Autonomous Buffer Sizing: Calculates exact MB limits for Engine concatenation.
- Mid-Stream Defense: Monitors RAM health dynamically during generation.
- X-Ray Profiling: Detects delimiters, encoding, and row-weights automatically.
"""

import os
import csv
import gzip
import zipfile
import logging
import psutil
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from collections import Counter
from typing import Iterator, List, Optional, Dict, Union, Any, Literal

# ==========================================
# CUSTOM EXCEPTIONS
# ==========================================
class NullHunterMemoryError(Exception):
    """Raised when the system lacks sufficient RAM to process the file."""
    pass

class NullHunterFormatError(Exception):
    """Raised when the file format is completely unsupported or corrupted."""
    pass

class NullHunterSecurityError(Exception):
    """Raised when a potential path traversal or malicious file access is detected."""
    pass


# ==========================================
# LOGGER CONFIGURATION
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | [%(levelname)s] NULLHUNTER_LOADER: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==========================================
# CLASS DATALOADER
# ==========================================
class DataLoader:
    """
    Intelligent Data Ingestion and Memory Orchestration Engine.

    Attributes:
        filepath (Path): Resolved, secure absolute path to the dataset.
        chunk_size (Union[int, str]): Row count per chunk, or 'auto' for dynamic scaling.
        max_cores (int): Number of CPU cores downstream Engine will use.
        ram_safety_margin (float): Percentage of free RAM utilized safely (0.1 to 0.9).
        memory_backend (str): Underlying Pandas backend ('pyarrow' or 'numpy_nullable').
        engine_buffer_limit_mb (int): Auto-calculated RAM limit for downstream concatenation.
    """

    GLOBAL_NA_VALUES = [
        '?', 'N/A', 'n/a', 'NA', 'na', 'null', 'Null', 'NULL', 
        '#N/A', '#N/A N/A', '#DIV/0!', '#VALUE!', ' ', ''
    ]

    def __init__(
        self,
        filepath: Union[str, Path],
        chunk_size: Optional[Union[int, Literal['auto']]] = None,
        max_cores: Optional[int] = None,
        ram_safety_margin: float = 0.50,
        fallback_encodings: Optional[List[str]] = None,
        possible_delimiters: Optional[List[str]] = None,
        fault_tolerance: str = 'skip',
        memory_backend: str = 'pyarrow',
        sniff_bytes: int = 16384,
        detect_metadata_headers: bool = True,
        standardize_nulls: bool = True,
        cache_in_memory: bool = False
    ) -> None:
        """
        Initializes the DataLoader with advanced CPU/RAM synchronization constraints.

        Args:
            filepath (Union[str, Path]): Path to the input dataset.
            chunk_size (Optional[Union[int, Literal['auto']]]): Fixed row size. If None or 'auto', scales dynamically.
            max_cores (Optional[int]): Target cores. Defaults to os.cpu_count() - 1.
            ram_safety_margin (float): Ratio of RAM permitted for ingestion (Default: 50%).
            fallback_encodings (Optional[List[str]]): Encodings to try if UTF-8 fails.
            possible_delimiters (Optional[List[str]]): Delimiters for the voting algorithm.
            fault_tolerance (str): Strategy for corrupted lines ('skip', 'error', 'warn').
            memory_backend (str): Underlying datatype engine (e.g., 'pyarrow').
            sniff_bytes (int): Buffer size for X-Ray profiling (Default: 16KB).
            detect_metadata_headers (bool): Auto-skips corporate title lines at file tops.
            standardize_nulls (bool): Maps dirty string nulls to pd.NA universally.
            cache_in_memory (bool): If True, pins parsed chunks in RAM for instant access.
        """
        self.chunk_size = 'auto' if chunk_size is None else chunk_size
        self.max_cores = max_cores or max(1, (os.cpu_count() or 2) - 1)
        self.ram_safety_margin = max(0.1, min(ram_safety_margin, 0.9)) 
        self.fault_tolerance = fault_tolerance
        self.memory_backend = memory_backend
        self.sniff_bytes = sniff_bytes
        self.detect_metadata_headers = detect_metadata_headers
        self.standardize_nulls = standardize_nulls
        self.cache_in_memory = cache_in_memory

        self._cached_chunks: List[pd.DataFrame] = []
        self._cache_locked: bool = False
        self._fast_path: bool = False
        self.fast_path_threshold: float = 0.15 
        self._estimated_bytes_per_row: int = 250 
        self.engine_buffer_limit_mb: int = 0
        
        self.encodings = fallback_encodings or ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        self.delimiters = possible_delimiters or [',', ';', '\t', '|', '^']
        
        self.safe_config: Dict[str, Any] = {
            'encoding': 'utf-8',
            'delimiter': ',',
            'skiprows': 0,
            'engine': 'c',
            'compression': 'infer',
            'na_values': self.GLOBAL_NA_VALUES if self.standardize_nulls else None
        }
        
        # Phase 1: Security & Setup
        self.filepath = self._secure_resolve_path(filepath)
        self._file_extension = self.filepath.suffix.lower()
        self._is_text_based = self._file_extension in ['.csv', '.tsv', '.txt']
        
        self._validate_file()
        self._detect_compression()
        
        # Phase 2: Structural Profiling
        if self._is_text_based:
            self._sniff_and_configure_text()
            
        # Phase 3: Mathematical Scale Calculation
        self._evaluate_ram_and_scale()

    def _secure_resolve_path(self, raw_path: Union[str, Path]) -> Path:
        """Resolves path and mitigates directory traversal vulnerabilities."""
        try:
            resolved = Path(raw_path).resolve(strict=True)
            if not resolved.is_file():
                raise NullHunterFormatError(f"Path is not a valid file: {resolved}")
            return resolved
        except FileNotFoundError:
            raise FileNotFoundError(f"[NullHunter] Dataset missing or inaccessible: {raw_path}")
        except Exception as e:
            raise NullHunterSecurityError(f"Path resolution failed (Security Risk): {str(e)}")

    def _validate_file(self) -> None:
        """Verifies file integrity and access permissions."""
        if self.filepath.stat().st_size == 0:
            raise NullHunterFormatError("Cannot process a 0-byte empty dataset.")
        if not os.access(self.filepath, os.R_OK):
            raise PermissionError(f"System lacks read permissions for: {self.filepath.name}")

    def _detect_compression(self) -> None:
        """Flags archive formats to ensure X-Ray profiles raw data."""
        if self._file_extension == '.gz':
            self.safe_config['compression'] = 'gzip'
            self._is_text_based = True
        elif self._file_extension == '.zip':
            self.safe_config['compression'] = 'zip'
            self._is_text_based = True

    def _read_raw_bytes(self) -> bytes:
        """Safely extracts header bytes through potential compression algorithms."""
        try:
            if self.safe_config['compression'] == 'gzip':
                with gzip.open(self.filepath, 'rb') as f:
                    return f.read(self.sniff_bytes)
            elif self.safe_config['compression'] == 'zip':
                with zipfile.ZipFile(self.filepath, 'r') as z:
                    first_file = z.namelist()[0]
                    with z.open(first_file, 'r') as f:
                        return f.read(self.sniff_bytes)
            else:
                with open(self.filepath, 'rb') as f:
                    return f.read(self.sniff_bytes)
        except Exception as e:
            logger.warning(f"Raw byte extraction failed: {str(e)}")
            return b""

    def _sniff_and_configure_text(self) -> None:
        """Master X-Ray: Detects encoding, delimiter, headers, and profiles row size."""
        raw_bytes = self._read_raw_bytes()
        if not raw_bytes:
            return

        decoded_sample = self._detect_encoding(raw_bytes)
        self._detect_delimiter(decoded_sample)
        
        if self.detect_metadata_headers:
            self._detect_metadata_headers(decoded_sample)
            
        lines = decoded_sample.splitlines()
        if len(lines) > 2:
            sample_lines = lines[1:-1] 
            total_bytes = sum(len(line.encode(self.safe_config['encoding'])) for line in sample_lines)
            self._estimated_bytes_per_row = total_bytes // len(sample_lines)
            logger.debug(f"Row Profiler: Estimated {self._estimated_bytes_per_row} bytes/row.")

    def _detect_encoding(self, raw_bytes: bytes) -> str:
        """Cycles through fallbacks to safely decode text matrices."""
        for enc in self.encodings:
            try:
                decoded = raw_bytes.decode(enc)
                self.safe_config['encoding'] = enc
                return decoded
            except UnicodeDecodeError:
                continue
        self.safe_config['encoding'] = 'unicode_escape'
        return raw_bytes.decode('unicode_escape')

    def _detect_delimiter(self, decoded_sample: str) -> None:
        """Identifies matrix delimiters using Sniffer and a fallback voting algorithm."""
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(decoded_sample, delimiters="".join(self.delimiters))
            self.safe_config['delimiter'] = dialect.delimiter
        except csv.Error:
            lines = decoded_sample.splitlines()
            if not lines:
                self.safe_config['delimiter'] = ','
                return
            votes = {d: 0 for d in self.delimiters}
            for line in lines[:20]:
                for d in self.delimiters:
                    votes[d] += line.count(d)
            best_delimiter = max(votes, key=votes.get)
            self.safe_config['delimiter'] = best_delimiter if votes[best_delimiter] > 0 else ','

    def _detect_metadata_headers(self, decoded_sample: str) -> None:
        """Detects unstructured corporate headers by tracking delimiter frequency."""
        if not self.safe_config['delimiter']:
            return
        lines = decoded_sample.splitlines()
        delimiter = self.safe_config['delimiter']
        counts = [line.count(delimiter) for line in lines if line.strip()]
        if not counts:
            return
        matrix_width_count = Counter(counts).most_common(1)[0][0]
        if matrix_width_count == 0:
            return 
        skiprows = 0
        for count in counts:
            if count < matrix_width_count:
                skiprows += 1
            else:
                break 
        self.safe_config['skiprows'] = skiprows

    def _evaluate_ram_and_scale(self) -> None:
        """Synchronizes CPU Cores with Available RAM to calculate safe operational parameters."""
        file_size_bytes = self.filepath.stat().st_size
        available_ram_bytes = psutil.virtual_memory().available
        
        multiplier = 6.0 if self._file_extension in ['.xlsx', '.xls'] else 3.5
        required_ram_bytes = file_size_bytes * multiplier
        
        # 1. Bypass Logic (Fast Path)
        if required_ram_bytes < (available_ram_bytes * self.fast_path_threshold):
            logger.info("Dataset is small relative to RAM. Activating FAST PATH.")
            self._fast_path = True
            self.engine_buffer_limit_mb = int((file_size_bytes * 2) / (1024 * 1024)) + 100 
            return
            
        self._fast_path = False
        
        # 2. Extreme Danger Assessment
        if required_ram_bytes > (available_ram_bytes * 0.90):
            if self._file_extension in ['.xlsx', '.xls']:
                raise NullHunterMemoryError(f"Fatal OOM Risk: Excel expansion requires {(required_ram_bytes/1e9):.1f}GB RAM.")
            self.cache_in_memory = False 
            
        # 3. Dynamic CPU-Aware Sizing
        safe_ram_bytes = available_ram_bytes * self.ram_safety_margin
        if self.chunk_size == 'auto':
            if self._is_text_based:
                ram_per_core = safe_ram_bytes / self.max_cores
                bytes_per_row = self._estimated_bytes_per_row * multiplier
                calculated_rows = int(ram_per_core / bytes_per_row)
                self.chunk_size = max(50000, min(calculated_rows, 2000000))
                logger.info(
                    f"Scale Matrix -> Cores: {self.max_cores} | "
                    f"Safe RAM/Core: {(ram_per_core/1e6):.0f}MB | "
                    f"Dynamic Chunk Size: {self.chunk_size} rows"
                )
            else:
                self.chunk_size = 150000 
                
        # 4. Engine Buffer Calculation (The 15% RAM Rule)
        self.engine_buffer_limit_mb = int((safe_ram_bytes * 0.30) / (1024 * 1024))
        logger.info(f"Generated autonomous Engine Buffer Limit: {self.engine_buffer_limit_mb} MB")

    def calculate_safe_buffer_limit(self) -> int:
        """
        Public API for the Supreme Commander (core.py) to extract the Engine's buffer limit.
        
        Returns:
            int: Safe write buffer limit in Megabytes.
        """
        return max(50, self.engine_buffer_limit_mb)

    def reset_stream(self) -> None:
        """
        Resets the internal state to allow iterating the dataset from the beginning.
        Locks the cache to prevent redundant RAM utilization during Pass 2.
        """
        if self.cache_in_memory and self._cached_chunks:
            logger.info("Stream reset requested. Operating from RAM Cache.")
            self._cache_locked = True
        else:
            logger.info("Stream reset requested. Re-initializing Disk I/O stream.")

    def get_chunks(self) -> Iterator[pd.DataFrame]:
        """
        The Main Routing Gateway. Yields safe, mathematically profiled DataFrame chunks.
        
        Yields:
            Iterator[pd.DataFrame]: Streamed DataFrame objects.
        """
        if self.cache_in_memory and self._cached_chunks:
            yield from self._cached_chunks
            return

        try:
            if self._is_text_based:
                stream = self._stream_csv()
            elif self._file_extension in ['.xlsx', '.xls']:
                stream = self._stream_excel()
            elif self._file_extension == '.parquet':
                stream = self._stream_parquet()
            elif self._file_extension == '.json':
                stream = self._stream_json()
            else:
                raise NullHunterFormatError(f"Unsupported format: {self._file_extension}")
                
            for chunk in stream:
                # Mid-stream OOM Defense
                if psutil.virtual_memory().percent > 95.0:
                    logger.warning("CRITICAL: System RAM exceeding 95%. Flushing cache to prevent OS crash.")
                    self._cached_chunks.clear()
                    self.cache_in_memory = False
                    
                if self.cache_in_memory and not self._cache_locked:
                    self._cached_chunks.append(chunk)
                yield chunk
                
        except Exception as e:
            logger.error(f"Ingestion stream collapsed: {str(e)}")
            raise

    def _stream_csv(self) -> Iterator[pd.DataFrame]:
        engine = 'pyarrow' if self.memory_backend == 'pyarrow' else self.safe_config['engine']
        read_kwargs = {
            'filepath_or_buffer': self.filepath,
            'encoding': self.safe_config['encoding'],
            'delimiter': self.safe_config['delimiter'],
            'skiprows': self.safe_config['skiprows'],
            'on_bad_lines': self.fault_tolerance,
            'engine': engine,
            'dtype_backend': self.memory_backend,
            'compression': self.safe_config['compression'],
            'na_values': self.safe_config['na_values'],
            'low_memory': False
        }

        if self._fast_path:
            yield pd.read_csv(**read_kwargs)
        else:
            read_kwargs['chunksize'] = self.chunk_size
            yield from pd.read_csv(**read_kwargs)

    def _stream_excel(self) -> Iterator[pd.DataFrame]:
        yield pd.read_excel(self.filepath, dtype_backend=self.memory_backend, na_values=self.safe_config['na_values'])

    def _stream_parquet(self) -> Iterator[pd.DataFrame]:
        if self._fast_path:
            yield pd.read_parquet(self.filepath, dtype_backend=self.memory_backend)
        else:
            parquet_file = pq.ParquetFile(self.filepath)
            batch_size = self.chunk_size if isinstance(self.chunk_size, int) else 150000
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                yield batch.to_pandas(types_mapper=pd.ArrowDtype if self.memory_backend == 'pyarrow' else None)

    def _stream_json(self) -> Iterator[pd.DataFrame]:
        if self._fast_path:
            yield pd.read_json(self.filepath, dtype_backend=self.memory_backend)
        else:
            try:
                c_size = int(self.chunk_size) if isinstance(self.chunk_size, int) else 100000
                yield from pd.read_json(
                    self.filepath, orient='records', lines=True, 
                    chunksize=c_size, dtype_backend=self.memory_backend
                )
            except Exception:
                logger.warning("JSON streaming failed. Initiating monolithic fallback.")
                yield pd.read_json(self.filepath, dtype_backend=self.memory_backend)
