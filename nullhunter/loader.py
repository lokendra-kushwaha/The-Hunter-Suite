import os
import csv
import warnings
import psutil
import pandas as pd

class DataLoader:
    def __init__(self, 
                 filepath, 
                 chunk_size=100000, 
                 fallback_encodings=None, 
                 possible_delimiters=None, 
                 fault_tolerance='skip', 
                 memory_backend='pyarrow', 
                 sniff_bytes=4096,
                 detect_metadata_headers=True,
                 cache_in_memory=False):
        
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.fault_tolerance = fault_tolerance
        self.memory_backend = memory_backend
        self.sniff_bytes = sniff_bytes
        self.detect_metadata_headers = detect_metadata_headers
        self.cache_in_memory = cache_in_memory
        
        self._cached_chunks = [] 
        
        self.encodings = fallback_encodings or ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'unicode_escape']
        self.delimiters = possible_delimiters or [',', ';', '\t', '|']
        
        self.safe_config = {
            'encoding': None,
            'delimiter': None,
            'skiprows': 0,
            'engine': 'c'
        }
        
        # 1. First validate the file exists
        self._validate_file()
        
        # 2. NEW: Check if caching is mathematically safe before sniffing
        if self.cache_in_memory:
            self._check_ram_safety()
            
        # 3. Proceed with standard sniffing
        self._sniff_and_configure()

    def _check_ram_safety(self):
        """
        Calculates the physical file size and compares it against the system's 
        available RAM to prevent catastrophic Out-Of-Memory (OOM) crashes.
        Pandas DataFrames typically consume 2x to 3x the raw CSV size in memory.
        """
        # Get actual file size in bytes and convert to Gigabytes
        file_size_bytes = os.path.getsize(self.filepath)
        file_size_gb = file_size_bytes / (1024 ** 3)
        
        # Get live available system RAM in Gigabytes
        available_ram_bytes = psutil.virtual_memory().available
        available_ram_gb = available_ram_bytes / (1024 ** 3)
        
        # Heuristic: We need at least 3x the file size in free RAM to safely cache it
        required_ram_gb = file_size_gb * 3.0
        
        print(f"[NullHunter Safety] Checking Memory Limits...")
        print(f"|-- File Size: {file_size_gb:.2f} GB")
        print(f"|-- Available RAM: {available_ram_gb:.2f} GB")
        print(f"|-- Estimated RAM Required: {required_ram_gb:.2f} GB")
        
        # If the required RAM is greater than 80% of our available RAM, we override the user!
        if required_ram_gb > (available_ram_gb * 0.8):
            warnings.warn(
                f"\n[NullHunter ALARM] Critical Memory Risk Detected!\n"
                f"Caching {file_size_gb:.2f}GB dataset requires ~{required_ram_gb:.2f}GB RAM.\n"
                f"You only have {available_ram_gb:.2f}GB available.\n"
                f"System is overriding user command. Forcing 'cache_in_memory=False' to prevent OS crash."
            )
            # OVERRIDE THE DANGER
            self.cache_in_memory = False
        else:
            print("[NullHunter Safety] Memory check passed. RAM Caching is SAFE to proceed.")

    def _validate_file(self):
        """Checks if file exists and handles Format Mismatch Warnings (Edge Case 4)."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"[NullHunter] Fatal Error: Dataset not found at {self.filepath}")
        
        # Do not crash on extension mismatch, just warn the user.
        if not self.filepath.endswith('.csv'):
            warnings.warn(f"[NullHunter] Warning: File extension is not .csv. System will attempt dynamic sniffing.")

    def _sniff_and_configure(self):
        """Master X-Ray function to detect Encoding, Delimiter, and Skiprows."""
        raw_bytes = self._read_raw_bytes()
        
        # Step A: Detect Safe Encoding (Fixes Edge Case 1: UnicodeDecodeError)
        decoded_sample = self._detect_encoding(raw_bytes)
        
        # Step B: Detect True Delimiter (Fixes Edge Case 3: 1D Matrix Trap)
        self._detect_delimiter(decoded_sample)
        
        # Note: Step C (Metadata Header Detection) will be executed next.

    def _read_raw_bytes(self):
        """Reads only the first N bytes to prevent RAM bloat during sniffing."""
        with open(self.filepath, 'rb') as f:
            return f.read(self.sniff_bytes)

    def _detect_encoding(self, raw_bytes):
        """Iterates through the fallback list to find a non-crashing encoding."""
        for enc in self.encodings:
            try:
                decoded_text = raw_bytes.decode(enc)
                self.safe_config['encoding'] = enc
                print(f"[NullHunter Sniffer] Success: Encoding locked to '{enc}'")
                return decoded_text
            except UnicodeDecodeError:
                continue # If it crashes, silently move to the next encoding
                
        # The Ultimate Fallback: unicode_escape will force-decode almost anything
        final_fallback = 'unicode_escape'
        self.safe_config['encoding'] = final_fallback
        print(f"[NullHunter Sniffer] Warning: All standard encodings failed. Forced '{final_fallback}'.")
        return raw_bytes.decode(final_fallback)

    def _detect_delimiter(self, decoded_sample):
        """Utilizes python's csv.Sniffer to identify the actual matrix structure."""
        try:
            sniffer = csv.Sniffer()
            # Feed the decoded text and our list of possible delimiters
            dialect = sniffer.sniff(decoded_sample, delimiters="".join(self.delimiters))
            self.safe_config['delimiter'] = dialect.delimiter
            print(f"[NullHunter Sniffer] Success: Delimiter locked to '{dialect.delimiter}'")
        except csv.Error:
            # If the file is too messy to sniff, fallback to standard CSV
            self.safe_config['delimiter'] = ','
            print("[NullHunter Sniffer] Warning: Delimiter sniffing failed. Defaulting to ','.")

    def _detect_metadata_headers(self, decoded_sample):
        """
        Fixes Edge Case 5: The Metadata Trap.
        Corporate CSVs often have 3-5 lines of text (e.g., "Report Date: 2026") 
        before the actual dataset headers begin. This method uses a mathematical 
        heuristic (Delimiter Mode Frequency) to find the true starting row.
        """
        if not self.detect_metadata_headers or not self.safe_config['delimiter']:
            return

        lines = decoded_sample.splitlines()
        delimiter = self.safe_config['delimiter']
        
        # Count the number of delimiters in each line
        # A valid data row will have a consistent number of delimiters (columns - 1)
        delimiter_counts = [line.count(delimiter) for line in lines if line.strip()]
        
        if not delimiter_counts:
            return
            
        # The 'mode' (most frequent count) represents the actual width of the matrix
        from collections import Counter
        matrix_width_count = Counter(delimiter_counts).most_common(1)[0][0]
        
        if matrix_width_count == 0:
            return # Matrix has no columns or delimiter detection failed
            
        # Traverse from the top and skip rows until we hit the actual matrix width
        skiprows = 0
        for count in delimiter_counts:
            if count < matrix_width_count:
                skiprows += 1
            else:
                break # We found the true header row
                
        self.safe_config['skiprows'] = skiprows
        if skiprows > 0:
            print(f"[NullHunter Sniffer] Success: Metadata detected. Auto-skipping {skiprows} rows.")
        else:
            print(f"[NullHunter Sniffer] Clean matrix detected. No metadata to skip.")

    def get_chunks(self):
            """
            The Master Generator Engine with Smart RAM Caching.
            """
            # THE CACHE INTERCEPTOR:
            # If caching is enabled and chunks are already in RAM, serve them instantly!
            if self.cache_in_memory and len(self._cached_chunks) > 0:
                print("[NullHunter Loader] Serving chunks directly from Ultra-Fast RAM Cache...")
                for chunk_id, chunk in enumerate(self._cached_chunks, start=1):
                    yield chunk
                return # Exit the function, preventing disk read completely.

            # If data is not in cache, proceed with standard disk reading
            if self.detect_metadata_headers:
                raw_bytes = self._read_raw_bytes()
                decoded_sample = raw_bytes.decode(self.safe_config['encoding'], errors='replace')
                self._detect_metadata_headers(decoded_sample)

            print(f"\n[NullHunter Loader] Initiating Out-of-Core Data Ingestion...")
            
            chunk_iterator = pd.read_csv(
                self.filepath,
                chunksize=self.chunk_size,
                encoding=self.safe_config['encoding'],
                delimiter=self.safe_config['delimiter'],
                skiprows=self.safe_config['skiprows'],
                on_bad_lines=self.fault_tolerance,
                engine=self.safe_config['engine'],
                dtype_backend=self.memory_backend,
                low_memory=False
            )
            
            for chunk_id, chunk in enumerate(chunk_iterator, start=1):
                print(f"[NullHunter Loader] Yielding Chunk {chunk_id} to the Scanner...")
                
                # SAVING TO CACHE: 
                # If the user wants to cache, append the chunk to our internal list before yielding
                if self.cache_in_memory:
                    self._cached_chunks.append(chunk)
                    
                yield chunk