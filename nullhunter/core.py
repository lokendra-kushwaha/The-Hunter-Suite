from nullhunter.loader import DataLoader
from nullhunter.engine import _ExecutionEngine

class NullHunter:
    """
    The Supreme Facade Class. 
    This is the only object the user interacts with. It abstracts all complex
    backend routing, multiprocessing, and file sniffing logic.
    """
    def __init__(self):
        # Initializing the internal sub-systems
        self.dataloader = DataLoader
        
        # Exposing the Execution Engine with the loader passed as a reference
        self.execute = _ExecutionEngine(loader_ref=self.dataloader)