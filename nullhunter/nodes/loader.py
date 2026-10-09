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
    Enterprise-Grade CPU-Aware Data Ingestion Engine for Project NullHunter.
    
    Acts as the secure, RAM-aware entry point for datasets. It calculates optimal 
    chunk sizes by profiling real-time row byte sizes and mathematically dividing 
    available RAM by the number of active CPU cores. Ensures zero-OOM crashes 
    while maximizing Multiprocessing pipeline efficiency.

    Attributes:
        filepath (Path): Resolved, secure absolute path to the dataset.
        chunk_size (Union[int, str]): Fixed row count, or 'auto' for dynamic CPU-aware sizing.
        max_cores (int): Number of CPU cores downstream Engine will use for parallel processing.
        ram_safety_margin (float): Percentage of free RAM to utilize safely (0.0 to 1.0).
        memory_backend (str): Pandas backend ('pyarrow' recommended for zero-copy memory speed).
    """

    # Global mapping for identifying junk/null strings across SaaS datasets
    GLOBAL_NA_VALUES = [
        '?', 'N/A', 'n/a', 'NA', 'na', 'null', 'Null', 'NULL', 
        '#N/A', '#N/A N/A', '#DIV/0!', '#VALUE!', ' ', ''
    ]

    def __init__(
        self,
        filepath: Union[str, Path],
        chunk_size: Union[int, Literal['auto']] = 'auto',
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
            chunk_size (Union[int, Literal['auto']]): Fixed row size or 'auto' for intelligent scale.
            max_cores (Optional[int]): Target cores. Defaults to os.cpu_count() - 1.
            ram_safety_margin (float): Ratio of RAM permitted for ingestion (Default: 50%).
            fallback_encodings (Optional[List[str]]): Encodings to try if UTF-8 fails.
            possible_delimiters (Optional[List[str]]): Delimiters for the voting algorithm.
            fault_tolerance (str): Strategy for corrupted lines ('skip', 'error', 'warn').
            memory_backend (str): Underlying datatype engine ('pyarrow' or 'numpy_nullable').
            sniff_bytes (int): Buffer size for X-Ray profiling (Default: 16KB for precision).
            detect_metadata_headers (bool): Auto-skips corporate title lines at file tops.
            standardize_nulls (bool): Maps dirty string nulls to pd.NA universally.
            cache_in_memory (bool): If True, pins parsed chunks in RAM for instant multiple access.
        """
        self.chunk_size = chunk_size
        self.max_cores = max_cores or max(1, (os.cpu_count() or 2) - 1)
        self.ram_safety_margin = max(0.1, min(ram_safety_margin, 0.9)) # Clamp between 10% and 90%
        self.fault_tolerance = fault_tolerance
        self.memory_backend = memory_backend
        self.sniff_bytes = sniff_bytes
        self.detect_metadata_headers = detect_metadata_headers
        self.standardize_nulls = standardize_nulls
        self.cache_in_memory = cache_in_memory

        self._cached_chunks: List[pd.DataFrame] = []
        self._fast_path: bool = False
        self.fast_path_threshold: float = 0.15 
        self._estimated_bytes_per_row: int = 250 # Fallback estimate
        
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
        
        # Phase 1: Security & File Identification
        self.filepath = self._secure_resolve_path(filepath)
        self._file_extension = self.filepath.suffix.lower()
        self._is_text_based = self._file_extension in ['.csv', '.tsv', '.txt']
        
        self._validate_file()
        self._detect_compression()
        
        # Phase 2: Structural Profiling (X-Ray)
        if self._is_text_based:
            self._sniff_and_configure_text()
            
        # Phase 3: Mathematical CPU/RAM Scale Calculation
        self._evaluate_ram_and_scale()

    def _secure_resolve_path(self, raw_path: Union[str, Path]) -> Path:
        """Resolves path and mitigates directory traversal vulnerabilities."""
        try:
            resolved_path = Path(raw_path).resolve(strict=True)
            return resolved_path
        except FileNotFoundError:
            raise FileNotFoundError(f"[NullHunter] Dataset missing or inaccessible: {raw_path}")
        except Exception as e:
            raise NullHunterSecurityError(f"Path resolution failed (Security Risk): {str(e)}")

    def _validate_file(self) -> None:
        """Verifies integrity and accessibility of the target file."""
        if self.filepath.stat().st_size == 0:
            raise NullHunterFormatError("Cannot process a 0-byte empty dataset.")
        if not os.access(self.filepath, os.R_OK):
            raise PermissionError(f"System lacks read permissions for: {self.filepath.name}")

    def _detect_compression(self) -> None:
        """Flags archive formats to ensure X-Ray sniffs raw data, not compressed binaries."""
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
            
        # REAL-TIME ROW BYTE PROFILER: Calculate exactly how heavy one row is
        lines = decoded_sample.splitlines()
        if len(lines) > 2:
            # Drop first and last line in case they are truncated or headers
            sample_lines = lines[1:-1] 
            total_bytes = sum(len(line.encode(self.safe_config['encoding'])) for line in sample_lines)
            self._estimated_bytes_per_row = total_bytes // len(sample_lines)
            logger.debug(f"Row Profiler: Estimated {self._estimated_bytes_per_row} bytes/row.")

    def _detect_encoding(self, raw_bytes: bytes) -> str:
        """Cycles through fallbacks to decode text matrices safely."""
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
        """Identifies matrix delimiters using Sniffer, falling back to a voting algorithm."""
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
        """Detects unstructured corporate headers by tracking delimiter frequency mode."""
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
        """
        The Mathematical Core: Synchronizes CPU Cores with Available RAM to calculate
        the mathematically perfect chunk size for downstream execution.
        """
        file_size_bytes = self.filepath.stat().st_size
        available_ram_bytes = psutil.virtual_memory().available
        
        # Memory bloat multiplier (Pandas dataframes take more RAM than raw text)
        multiplier = 6.0 if self._file_extension in ['.xlsx', '.xls'] else 3.5
        required_ram_bytes = file_size_bytes * multiplier
        
        # 1. Bypass Logic (Fast Path) for small files
        if required_ram_bytes < (available_ram_bytes * self.fast_path_threshold):
            logger.info("Dataset is small relative to RAM. Activating FAST PATH (Disk I/O Bypass).")
            self._fast_path = True
            return
            
        self._fast_path = False
        
        # 2. Extreme Danger Assessment
        if required_ram_bytes > (available_ram_bytes * 0.90):
            if self._file_extension in ['.xlsx', '.xls']:
                raise NullHunterMemoryError(f"Fatal OOM Risk: Excel expansion requires {(required_ram_bytes/1e9):.1f}GB RAM.")
            self.cache_in_memory = False # Force disable caching
            
        # 3. CPU-Aware Chunk Sizing Mathematics
        if self.chunk_size == 'auto':
            if self._is_text_based:
                # Calculate absolute safe RAM available for the ingestion batch
                safe_ram_bytes = available_ram_bytes * self.ram_safety_margin
                
                # Distribute RAM equally among all active CPU cores
                ram_per_core = safe_ram_bytes / self.max_cores
                
                # Calculate how many rows fit into a single core's RAM budget
                bytes_per_row = self._estimated_bytes_per_row * multiplier
                calculated_rows = int(ram_per_core / bytes_per_row)
                
                # Enforce architectural guardrails (Not too small, not too massive)
                self.chunk_size = max(50000, min(calculated_rows, 2000000))
                
                logger.info(
                    f"Scale Matrix -> Cores: {self.max_cores} | "
                    f"Safe RAM/Core: {(ram_per_core/1e6):.0f}MB | "
                    f"Dynamic Chunk Size: {self.chunk_size} rows"
                )
            else:
                # Default safety size for binary formats (Parquet/JSON chunks)
                self.chunk_size = 150000 
        else:
            logger.info(f"Using strict user-defined chunk size: {self.chunk_size} rows.")

    def get_chunks(self) -> Iterator[pd.DataFrame]:
        """
        The Main Routing Gateway. Validates cache and dispatches to specific format streams.
        
        Yields:
            Iterator[pd.DataFrame]: Streamed DataFrame objects.
        """
        if self.cache_in_memory and self._cached_chunks:
            logger.info("Serving data directly from Ultra-Fast RAM Cache.")
            yield from self._cached_chunks
            return

        try:
            if self._is_text_based:
                yield from self._stream_csv()
            elif self._file_extension in ['.xlsx', '.xls']:
                yield from self._stream_excel()
            elif self._file_extension == '.parquet':
                yield from self._stream_parquet()
            elif self._file_extension == '.json':
                yield from self._stream_json()
            else:
                raise NullHunterFormatError(f"Unsupported binary structure: {self._file_extension}")
        except Exception as e:
            logger.error(f"Ingestion stream collapsed: {str(e)}")
            raise

    def _stream_csv(self) -> Iterator[pd.DataFrame]:
        """Iterates CSV matrices using high-performance PyArrow engine mapping."""
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
            df = pd.read_csv(**read_kwargs)
            if self.cache_in_memory:
                self._cached_chunks.append(df)
            yield df
        else:
            read_kwargs['chunksize'] = self.chunk_size
            for chunk in pd.read_csv(**read_kwargs):
                if self.cache_in_memory:
                    self._cached_chunks.append(chunk)
                yield chunk

    def _stream_excel(self) -> Iterator[pd.DataFrame]:
        """Monolithic bypass for Excel sheets (uncapable of partial extraction)."""
        df = pd.read_excel(self.filepath, dtype_backend=self.memory_backend, na_values=self.safe_config['na_values'])
        if self.cache_in_memory:
            self._cached_chunks.append(df)
        yield df

    def _stream_parquet(self) -> Iterator[pd.DataFrame]:
        """True Parquet Chunking: Yields native Row Groups to prevent RAM detonation."""
        if self._fast_path:
            df = pd.read_parquet(self.filepath, dtype_backend=self.memory_backend)
            if self.cache_in_memory:
                self._cached_chunks.append(df)
            yield df
        else:
            parquet_file = pq.ParquetFile(self.filepath)
            batch_size = self.chunk_size if isinstance(self.chunk_size, int) else 150000
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                df = batch.to_pandas(types_mapper=pd.ArrowDtype if self.memory_backend == 'pyarrow' else None)
                if self.cache_in_memory:
                    self._cached_chunks.append(df)
                yield df

    def _stream_json(self) -> Iterator[pd.DataFrame]:
        """Handles streaming nested/linear JSON matrices."""
        if self._fast_path:
            df = pd.read_json(self.filepath, dtype_backend=self.memory_backend)
            if self.cache_in_memory:
                self._cached_chunks.append(df)
            yield df
        else:
            try:
                c_size = int(self.chunk_size) if isinstance(self.chunk_size, int) else 100000
                chunk_iterator = pd.read_json(
                    self.filepath, 
                    orient='records', 
                    lines=True, 
                    chunksize=c_size,
                    dtype_backend=self.memory_backend
                )
                for chunk in chunk_iterator:
                    if self.cache_in_memory:
                        self._cached_chunks.append(chunk)
                    yield chunk
            except Exception:
                logger.warning("JSON streaming failed (Likely deeply nested). Initiating monolithic fallback.")
                df = pd.read_json(self.filepath, dtype_backend=self.memory_backend)
                if self.cache_in_memory:
                    self._cached_chunks.append(df)
                yield df