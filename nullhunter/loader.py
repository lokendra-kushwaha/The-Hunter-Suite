import csv
import gzip
import zipfile
import logging
import psutil
import pandas as pd
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Union, Any

# ==========================================
# CUSTOM EXCEPTIONS
# ==========================================
class NullHunterMemoryError(Exception):
    """Raised when the system lacks sufficient RAM to process the file."""
    pass

class NullHunterFormatError(Exception):
    """Raised when the file format is completely unsupported or corrupted."""
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
    Enterprise-Grade Data Ingestion Engine for Project NullHunter.
    
    This module acts as the entry point for all raw datasets. It automatically
    detects file formats, sniffs delimiters and encodings for text files, 
    evaluates RAM safety to prevent OOM (Out-Of-Memory) crashes, and implements 
    smart routing (Fast Path for small files vs. Chunk Path for massive out-of-core data).

    Attributes:
        filepath (Path): Absolute or relative path to the dataset.
        chunk_size (int): Number of rows to read per iteration (for supported formats).
        memory_backend (str): Pandas backend to use ('pyarrow' is highly recommended for RAM efficiency).
        cache_in_memory (bool): If True, retains all chunks in RAM for instant re-access.
        fast_path_threshold (float): Percentage of available RAM below which a file is loaded entirely (default: 0.15).
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        chunk_size: int = 100000,
        fallback_encodings: Optional[List[str]] = None,
        possible_delimiters: Optional[List[str]] = None,
        fault_tolerance: str = 'skip',
        memory_backend: str = 'pyarrow',
        sniff_bytes: int = 4096,
        detect_metadata_headers: bool = True,
        cache_in_memory: bool = False
    ) -> None:
        """
        Initializes the DataLoader with configuration and safety constraints.
        
        Args:
            filepath (Union[str, Path]): Path to the input data file.
            chunk_size (int): Row count per chunk for out-of-core processing.
            fallback_encodings (Optional[List[str]]): List of encodings to try during sniffing.
            possible_delimiters (Optional[List[str]]): List of delimiters to test.
            fault_tolerance (str): Strategy for bad lines ('skip', 'error', etc.).
            memory_backend (str): DataFrame backend ('pyarrow' or 'numpy_nullable').
            sniff_bytes (int): Number of bytes to read for metadata detection.
            detect_metadata_headers (bool): Whether to auto-skip junk corporate headers.
            cache_in_memory (bool): If True, saves all chunks to RAM (if safe).
        """
        
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self.fault_tolerance = fault_tolerance
        self.memory_backend = memory_backend
        self.sniff_bytes = sniff_bytes
        self.detect_metadata_headers = detect_metadata_headers
        self.cache_in_memory = cache_in_memory

        # Internal state and buffers
        self._cached_chunks: List[pd.DataFrame] = []
        self._fast_path: bool = False
        self.fast_path_threshold: float = 0.15  # Bypass chunking if required RAM < 15% of free RAM
        
        self.encodings = fallback_encodings or ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'unicode_escape']
        self.delimiters = possible_delimiters or [',', ';', '\t', '|']
        
        self.safe_config: Dict[str, Any] = {
            'encoding': 'utf-8',
            'delimiter': ',',
            'skiprows': 0,
            'engine': 'c',
            'compression': 'infer'
        }
        
        self._file_extension = self.filepath.suffix.lower()
        self._is_text_based = self._file_extension in ['.csv', '.tsv', '.txt']
        
        # Initialization Pipeline
        self._validate_file()
        self._detect_compression()
        
        # Always run RAM safety check to determine Fast Path vs Chunk Path
        self._check_ram_safety()

        if self._is_text_based:
            self._sniff_and_configure_text()


    def _validate_file(self) -> None:
        """Validates file existence and structural integrity."""
        if not self.filepath.exists():
            logger.error(f"Fatal Error: Dataset not found at {self.filepath}")
            raise FileNotFoundError(f"[NullHunter] Dataset not found at {self.filepath}")
        
        if self.filepath.stat().st_size == 0:
            logger.error("The provided file is completely empty (0 bytes).")
            raise NullHunterFormatError("Cannot process an empty file.")
            
        logger.info(f"File validated: {self.filepath.name} ({self._file_extension})")


    def _detect_compression(self) -> None:
        """Detects if the CSV/TXT file is compressed to handle sniffing properly."""
        if self._file_extension == '.gz':
            self.safe_config['compression'] = 'gzip'
            self._is_text_based = True
            logger.info("Detected GZIP compression.")
        elif self._file_extension == '.zip':
            self.safe_config['compression'] = 'zip'
            self._is_text_based = True
            logger.info("Detected ZIP compression.")


    def _check_ram_safety(self) -> None:
        """
        Evaluates physical file size against available RAM to set ingestion paths.
        Activates Fast Path for small files and applies safety stops for massive files.
        
        Raises:
            NullHunterMemoryError: If caching is forced and RAM is vastly insufficient.
        """
        file_size_bytes = self.filepath.stat().st_size
        file_size_gb = file_size_bytes / (1024 ** 3)
        
        available_ram_bytes = psutil.virtual_memory().available
        available_ram_gb = available_ram_bytes / (1024 ** 3)
        
        # Excel inflates massively; CSV inflates moderately.
        multiplier = 6.0 if self._file_extension in ['.xlsx', '.xls'] else 3.0
        required_ram_gb = file_size_gb * multiplier
        
        logger.info(f"RAM Check -> File: {file_size_gb:.3f}GB | Required: ~{required_ram_gb:.2f}GB | Available: {available_ram_gb:.2f}GB")
        
        # 1. Extreme Danger (OOM Risk)
        if required_ram_gb > (available_ram_gb * 0.85):
            if self._file_extension in ['.xlsx', '.xls']:
                raise NullHunterMemoryError(
                    f"Out of Memory Risk: Loading this Excel file requires ~{required_ram_gb:.2f}GB RAM, "
                    f"but only {available_ram_gb:.2f}GB is available."
                )
            else:
                logger.warning("Critical Memory Risk! Forcing Chunk Path and disabling memory caching.")
                self.cache_in_memory = False
                self._fast_path = False
                
        # 2. Fast Path (Abundant RAM / Small File)
        elif required_ram_gb < (available_ram_gb * self.fast_path_threshold):
            logger.info("Small dataset detected relative to RAM. Activating FAST PATH (Disk I/O Bypass).")
            self._fast_path = True
            
        # 3. Standard Path (Chunking required)
        else:
            logger.info("RAM Check Passed. Using standard Chunk Path for safe out-of-core ingestion.")
            self._fast_path = False


    def _read_raw_bytes(self) -> bytes:
        """
        Reads the first N bytes safely, handling compression transparently.
        
        Returns:
            bytes: The raw byte sequence from the file header.
        """
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
            logger.error(f"Failed to read raw bytes: {str(e)}")
            return b""


    def _sniff_and_configure_text(self) -> None:
        """Master X-Ray function for text files. Detects encoding, delimiter, and headers."""
        raw_bytes = self._read_raw_bytes()
        if not raw_bytes:
            return

        decoded_sample = self._detect_encoding(raw_bytes)
        self._detect_delimiter(decoded_sample)
        
        if self.detect_metadata_headers:
            self._detect_metadata_headers(decoded_sample)


    def _detect_encoding(self, raw_bytes: bytes) -> str:
        """Iterates through fallback encodings to find a non-crashing format."""
        for enc in self.encodings:
            try:
                decoded_text = raw_bytes.decode(enc)
                self.safe_config['encoding'] = enc
                logger.debug(f"Encoding locked to '{enc}'")
                return decoded_text
            except UnicodeDecodeError:
                continue
                
        final_fallback = 'unicode_escape'
        self.safe_config['encoding'] = final_fallback
        logger.warning(f"All standard encodings failed. Forced '{final_fallback}'.")
        return raw_bytes.decode(final_fallback)


    def _detect_delimiter(self, decoded_sample: str) -> None:
        """Utilizes Python's csv.Sniffer to identify the matrix structure."""
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(decoded_sample, delimiters="".join(self.delimiters))
            self.safe_config['delimiter'] = dialect.delimiter
            logger.info(f"Delimiter locked to '{dialect.delimiter}'")
        except csv.Error:
            self.safe_config['delimiter'] = ','
            logger.warning("Delimiter sniffing failed. Defaulting to ','.")


    def _detect_metadata_headers(self, decoded_sample: str) -> None:
        """Detects junk corporate headers by finding the mode of delimiter frequencies."""
        if not self.safe_config['delimiter']:
            return

        lines = decoded_sample.splitlines()
        delimiter = self.safe_config['delimiter']
        
        delimiter_counts = [line.count(delimiter) for line in lines if line.strip()]
        if not delimiter_counts:
            return
            
        from collections import Counter
        matrix_width_count = Counter(delimiter_counts).most_common(1)[0][0]
        
        if matrix_width_count == 0:
            return 
            
        skiprows = 0
        for count in delimiter_counts:
            if count < matrix_width_count:
                skiprows += 1
            else:
                break 
                
        self.safe_config['skiprows'] = skiprows
        if skiprows > 0:
            logger.info(f"Metadata trap detected. Auto-skipping {skiprows} rows.")


    def get_chunks(self) -> Iterator[pd.DataFrame]:
        """
        The Master Generator Engine. 
        Routes data reading based on file format, memory profile, and bypass flags.
        
        Yields:
            Iterator[pd.DataFrame]: Data chunks (or full dataframe if Fast Path is active).
        """
        if self.cache_in_memory and self._cached_chunks:
            logger.info("Serving chunks directly from Ultra-Fast RAM Cache...")
            for chunk in self._cached_chunks:
                yield chunk
            return

        logger.info("Initiating Data Ingestion Routing...")

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
                raise NullHunterFormatError(f"Unsupported file format: {self._file_extension}")
        except Exception as e:
            logger.error(f"Ingestion failed during processing: {str(e)}")
            raise


    def _stream_csv(self) -> Iterator[pd.DataFrame]:
        """Generates chunks for text files, honoring the Fast Path bypass."""
        if self._fast_path:
            logger.info("[Fast Path] Bypassing chunking. Loading entire CSV into memory.")
            df = pd.read_csv(
                self.filepath,
                encoding=self.safe_config['encoding'],
                delimiter=self.safe_config['delimiter'],
                skiprows=self.safe_config['skiprows'],
                on_bad_lines=self.fault_tolerance,
                engine=self.safe_config['engine'],
                dtype_backend=self.memory_backend,
                compression=self.safe_config['compression'],
                low_memory=False
            )
            if self.cache_in_memory:
                self._cached_chunks.append(df)
            yield df
        else:
            logger.info("[Chunk Path] Streaming CSV from disk sequentially.")
            chunk_iterator = pd.read_csv(
                self.filepath,
                chunksize=self.chunk_size,
                encoding=self.safe_config['encoding'],
                delimiter=self.safe_config['delimiter'],
                skiprows=self.safe_config['skiprows'],
                on_bad_lines=self.fault_tolerance,
                engine=self.safe_config['engine'],
                dtype_backend=self.memory_backend,
                compression=self.safe_config['compression'],
                low_memory=False
            )
            
            for chunk_id, chunk in enumerate(chunk_iterator, start=1):
                logger.debug(f"Yielding Text Chunk {chunk_id} to the Scanner...")
                if self.cache_in_memory:
                    self._cached_chunks.append(chunk)
                yield chunk


    def _stream_excel(self) -> Iterator[pd.DataFrame]:
        """Excel files do not support chunking well. Loaded wholly."""
        logger.info("[Fast Path] Excel format detected. Loading entire file into memory.")
        df = pd.read_excel(
            self.filepath,
            dtype_backend=self.memory_backend
        )
        if self.cache_in_memory:
            self._cached_chunks.append(df)
        yield df


    def _stream_parquet(self) -> Iterator[pd.DataFrame]:
        """Parquet files are natively optimized and loaded directly."""
        logger.info("[Fast Path] Parquet format detected. Loading optimized binary matrix.")
        df = pd.read_parquet(
            self.filepath,
            dtype_backend=self.memory_backend
        )
        if self.cache_in_memory:
            self._cached_chunks.append(df)
        yield df


    def _stream_json(self) -> Iterator[pd.DataFrame]:
        """Generates chunks for JSON files, honoring the Fast Path bypass."""
        if self._fast_path:
            logger.info("[Fast Path] Bypassing chunking. Loading entire JSON into memory.")
            df = pd.read_json(self.filepath, dtype_backend=self.memory_backend)
            if self.cache_in_memory:
                self._cached_chunks.append(df)
            yield df
        else:
            logger.info("[Chunk Path] Attempting JSON lines streaming.")
            try:
                chunk_iterator = pd.read_json(
                    self.filepath, 
                    orient='records', 
                    lines=True, 
                    chunksize=self.chunk_size,
                    dtype_backend=self.memory_backend
                )
                for chunk in chunk_iterator:
                    if self.cache_in_memory:
                        self._cached_chunks.append(chunk)
                    yield chunk
            except Exception:
                logger.warning("JSON streaming failed. Falling back to monolithic load.")
                df = pd.read_json(self.filepath, dtype_backend=self.memory_backend)
                if self.cache_in_memory:
                    self._cached_chunks.append(df)
                yield df