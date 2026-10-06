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


class DataLoader:
    """
    Enterprise-Grade Data Ingestion Engine for Project NullHunter.
    
    This module acts as the entry point for all raw datasets. It automatically
    detects file formats, sniffs delimiters and encodings for text files, 
    evaluates RAM safety to prevent OOM (Out-Of-Memory) crashes, and streams 
    data into chunks for out-of-core processing.

    Attributes:
        filepath (Path): Absolute or relative path to the dataset.
        chunk_size (int): Number of rows to read per iteration (for supported formats).
        memory_backend (str): Pandas backend to use ('pyarrow' is highly recommended for RAM efficiency).
        cache_in_memory (bool): If True, retains all chunks in RAM for instant re-access.
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
        """Initializes the DataLoader with configuration and safety constraints."""
        
        self.filepath = Path(filepath)
        self.chunk_size = chunk_size
        self.fault_tolerance = fault_tolerance
        self.memory_backend = memory_backend
        self.sniff_bytes = sniff_bytes
        self.detect_metadata_headers = detect_metadata_headers
        self.cache_in_memory = cache_in_memory

        self._cached_chunks: List[pd.DataFrame] = []
        
        self.encodings = fallback_encodings or ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'unicode_escape']
        self.delimiters = possible_delimiters or [',', ';', '\t', '|']
        
        # State configuration for text-based files
        self.safe_config: Dict[str, Any] = {
            'encoding': 'utf-8',
            'delimiter': ',',
            'skiprows': 0,
            'engine': 'c',
            'compression': 'infer'
        }
        
        self._file_extension = self.filepath.suffix.lower()
        self._is_text_based = self._file_extension in ['.csv', '.tsv', '.txt']
        
        self._validate_file()
        self._detect_compression()
        
        if self.cache_in_memory or not self._is_text_based:
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
            self._is_text_based = True # Assuming it's a zipped csv
            logger.info("Detected GZIP compression.")
        elif self._file_extension == '.zip':
            self.safe_config['compression'] = 'zip'
            self._is_text_based = True
            logger.info("Detected ZIP compression.")


    def _check_ram_safety(self) -> None:
        """
        Calculates physical file size and compares it against system's available RAM.
        Applies dynamic heuristics based on the file format (Excel bloats more than CSV).
        
        Raises:
            NullHunterMemoryError: If caching is forced and RAM is vastly insufficient.
        """
        file_size_bytes = self.filepath.stat().st_size
        file_size_gb = file_size_bytes / (1024 ** 3)
        
        available_ram_bytes = psutil.virtual_memory().available
        available_ram_gb = available_ram_bytes / (1024 ** 3)
        
        # Heuristics: Excel uses ~6x its size in RAM. CSV uses ~3x.
        multiplier = 6.0 if self._file_extension in ['.xlsx', '.xls'] else 3.0
        required_ram_gb = file_size_gb * multiplier
        
        logger.info(f"RAM Check -> File: {file_size_gb:.3f}GB | Required: ~{required_ram_gb:.2f}GB | Available: {available_ram_gb:.2f}GB")
        
        if required_ram_gb > (available_ram_gb * 0.85):
            if self._file_extension in ['.xlsx', '.xls']:
                # Excel cannot be chunked easily. If it doesn't fit, we must crash early.
                raise NullHunterMemoryError(
                    f"Out of Memory Risk: Loading this Excel file requires ~{required_ram_gb:.2f}GB RAM, "
                    f"but only {available_ram_gb:.2f}GB is available. Consider converting to CSV."
                )
            else:
                logger.warning("Critical Memory Risk! Overriding user cache command to prevent OS crash.")
                self.cache_in_memory = False
        else:
            logger.info("RAM Check Passed. Caching / Bulk Loading is safe.")


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
                    # Sniff the first file in the zip archive
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
        """Master X-Ray function for text files (CSV, TSV). Detects encoding, delimiter, and headers."""
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
        Routes data reading based on file format and yields DataFrames.
        
        Yields:
            Iterator[pd.DataFrame]: Data chunks for processing.
        """
        # Cache Interceptor
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
        """Internal generator for streaming text-based files."""
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
        """
        Internal generator for Excel files. 
        Note: Pandas cannot chunk Excel files easily, so it is loaded wholly.
        """
        logger.warning("Excel format detected. Chunking is disabled; loading entire file into memory.")
        df = pd.read_excel(
            self.filepath,
            dtype_backend=self.memory_backend
        )
        if self.cache_in_memory:
            self._cached_chunks.append(df)
        yield df


    def _stream_parquet(self) -> Iterator[pd.DataFrame]:
        """Internal generator for Parquet files (Highly Optimized)."""
        logger.info("Parquet format detected. Loading optimized binary matrix.")
        # Parquet natively supports row groups, but for simplicity we load full or use pyarrow dataset iterators
        df = pd.read_parquet(
            self.filepath,
            dtype_backend=self.memory_backend
        )
        if self.cache_in_memory:
            self._cached_chunks.append(df)
        yield df


    def _stream_json(self) -> Iterator[pd.DataFrame]:
        """Internal generator for JSON files. Attempts lines=True streaming first."""
        logger.info("JSON format detected. Attempting to load...")
        try:
            # Try chunked streaming for JSON-lines
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
            # Fallback to loading whole JSON if it's a monolithic block
            logger.warning("JSON streaming failed. Loading monolithic JSON block.")
            df = pd.read_json(self.filepath, dtype_backend=self.memory_backend)
            if self.cache_in_memory:
                self._cached_chunks.append(df)
            yield df