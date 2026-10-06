import os
import csv
import warnings
import pandas as pd

class DataLoader:
    """
    NullHunter's Enterprise-Grade Data Ingestion Module (The File Sniffer).
    Dynamically handles encoding crashes, delimiter illusions, metadata traps,
    and memory fragmentation before the data even touches the RAM.
    """
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
        
        # 1. Parameter Assignment
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.fault_tolerance = fault_tolerance
        self.memory_backend = memory_backend
        self.sniff_bytes = sniff_bytes
        self.detect_metadata_headers = detect_metadata_headers
        self.cache_in_memory = cache_in_memory

        # New: Internal RAM Cache Storage
        self._cached_chunks = []
        
        # 2. Smart Defaults for Core Edge Cases
        # List of encodings to try before giving up (Edge Case 1)
        self.encodings = fallback_encodings or ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'unicode_escape']
        # List of potential delimiters to sniff (Edge Case 3)
        self.delimiters = possible_delimiters or [',', ';', '\t', '|']
        
        # 3. The Safe Configuration Dictionary
        # This will be dynamically populated by the Sniffer before loading data
        self.safe_config = {
            'encoding': None,
            'delimiter': None,
            'skiprows': 0,
            'engine': 'c'  # Default pandas C-engine for speed
        }
        
        # 4. Trigger the Pre-Load Sniffing Protocol immediately
        self._validate_file()
        self._sniff_and_configure()

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