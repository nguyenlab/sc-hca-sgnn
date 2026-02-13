import json
from pathlib import Path
from typing import Dict, Any, Optional


class InjectedContractViewer:
    """
    Viewer for inspecting injected vulnerabilities in contracts.
    
    Loads metadata and displays injected code regions with context.
    """
    
    ANSI_RED = "\033[91m"
    ANSI_RESET = "\033[0m"
    
    def __init__(self, metadata_path: str | Path):
        """
        Initialize viewer with metadata file.
        
        Args:
            metadata_path: Path to the metadata JSON file
        """
        self.metadata_path = Path(metadata_path)
        self.metadata: Optional[Dict[str, Any]] = None
        self.contract_code: Optional[bytes] = None
    
    def load(self) -> None:
        """Load metadata and contract content."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        with open(self.metadata_path) as f:
            self.metadata = json.load(f)
        
        contract_path = Path(self.metadata['output_contract'])
        if not contract_path.exists():
            raise FileNotFoundError(f"Contract file not found: {contract_path}")
        
        with open(contract_path, 'rb') as f:
            self.contract_code = f.read()
    
    def display_with_context(self, context_lines: int = 3) -> None:
        """
        Display injected code regions with surrounding context.
        
        Args:
            context_lines: Number of context lines before/after
        """
        if self.metadata is None or self.contract_code is None:
            self.load()
        
        self._print_header()
        
        if not self.metadata['injected_regions']:
            print("No injected regions found in metadata.")
            return
        
        print(f"Injected Regions: {len(self.metadata['injected_regions'])}")
        print("=" * 80)
        print()
        
        for i, region in enumerate(self.metadata['injected_regions'], 1):
            self._display_region(i, region, context_lines)
    
    def display_code_only(self) -> None:
        """Display only the injected code without context."""
        if self.metadata is None or self.contract_code is None:
            self.load()
        
        print("=" * 80)
        print("INJECTED CODE ONLY")
        print("=" * 80)
        print()
        
        for i, region in enumerate(self.metadata['injected_regions'], 1):
            start_byte = region['start_byte']
            end_byte = region['end_byte']
            component = region['component']
            
            code = self.contract_code[start_byte:end_byte].decode('utf-8', errors='ignore')
            
            print(f"[{i}] {component.upper()}:")
            print(code)
            print()
    
    def _print_header(self) -> None:
        """Print the viewer header with metadata info."""
        print("=" * 80)
        print("INJECTED VULNERABILITY VIEW")
        print("=" * 80)
        print()
        print(f"Source: {self.metadata['source_contract']}")
        print(f"Output: {self.metadata['output_contract']}")
        print(f"Vulnerability Type: {self.metadata['vulnerability_type']}")
        print(f"Injection Mode: {self.metadata['injection_mode']}")
        print(f"Template: {self.metadata['template_name']}")
        print(f"Solidity Version: {self.metadata['solidity_version']}")
        print()
    
    def _display_region(self, index: int, region: Dict, context_lines: int) -> None:
        """Display a single injected region with context."""
        start_byte = region['start_byte']
        end_byte = region['end_byte']
        component = region['component']
        description = region.get('description', '')
        
        # Extract injected code
        injected_code = self.contract_code[start_byte:end_byte].decode('utf-8', errors='ignore')
        
        # Get context
        context_before = self._get_context_before(start_byte, context_lines)
        context_after = self._get_context_after(end_byte, context_lines)
        
        # Print region info
        print(f"Region {index}: {component.upper()}")
        print(f"  Bytes: {start_byte}-{end_byte} ({end_byte - start_byte} bytes)")
        print(f"  Description: {description}")
        print("-" * 80)
        
        # Display with context
        if context_before.strip():
            print(context_before, end='')
        
        # Highlight injected code in red
        print(f"{self.ANSI_RED}{injected_code}{self.ANSI_RESET}", end='')
        
        if context_after.strip():
            print(context_after)
        
        print()
        print("-" * 80)
        print()
    
    def _get_context_before(self, start_byte: int, num_lines: int) -> str:
        """Get context lines before a position."""
        context_start = max(0, start_byte - 200)
        context = self.contract_code[context_start:start_byte].decode('utf-8', errors='ignore')
        
        if '\n' in context:
            context = '\n'.join(context.split('\n')[-num_lines:])
        
        return context
    
    def _get_context_after(self, end_byte: int, num_lines: int) -> str:
        """Get context lines after a position."""
        context_end = min(len(self.contract_code), end_byte + 200)
        context = self.contract_code[end_byte:context_end].decode('utf-8', errors='ignore')
        
        if '\n' in context:
            context = '\n'.join(context.split('\n')[:num_lines])
        
        return context
