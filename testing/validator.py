import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

from .compiler import SolidityCompiler, CompilationResult


@dataclass
class ValidationStats:
    """Statistics from contract validation."""
    total: int = 0
    success: int = 0
    failed: int = 0
    by_type: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0}))
    by_template: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {'total': 0, 'success': 0, 'failed': 0}))
    failed_contracts: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.success / self.total * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'summary': {
                'total': self.total,
                'success': self.success,
                'failed': self.failed,
                'success_rate': f"{self.success_rate:.2f}%"
            },
            'by_type': dict(self.by_type),
            'by_template': dict(self.by_template),
            'failed_contracts': self.failed_contracts
        }


class ContractValidator:
    """
    Service for validating injected smart contracts.
    
    Tests contracts for compilation success and collects statistics
    by injection type and template.
    """
    
    def __init__(self, compiler: Optional[SolidityCompiler] = None):
        """
        Initialize validator.
        
        Args:
            compiler: Optional compiler instance (creates new one if not provided)
        """
        self.compiler = compiler or SolidityCompiler()
        self.stats = ValidationStats()
    
    def validate_directory(self, directory: str | Path) -> ValidationStats:
        """
        Validate all contracts in a directory.
        
        Args:
            directory: Path to directory containing .sol files
            
        Returns:
            ValidationStats with results
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory}")
        
        sol_files = list(directory.rglob('*.sol'))
        
        for sol_file in sol_files:
            self._validate_contract(sol_file)
        
        return self.stats
    
    def validate_contracts(self, contract_paths: List[Path]) -> ValidationStats:
        """
        Validate a list of contracts.
        
        Args:
            contract_paths: List of contract file paths
            
        Returns:
            ValidationStats with results
        """
        for path in contract_paths:
            self._validate_contract(path)
        
        return self.stats
    
    def _validate_contract(self, sol_file: Path) -> None:
        """Validate a single contract file."""
        self.stats.total += 1
        
        # Get injection info from metadata
        injection_type, template, _ = self._get_injection_info(str(sol_file))
        
        # Compile
        result = self.compiler.compile_file(sol_file)
        
        # Update statistics
        if result.success:
            self.stats.success += 1
            self.stats.by_type[injection_type]['success'] += 1
            self.stats.by_template[template]['success'] += 1
        else:
            self.stats.failed += 1
            self.stats.by_type[injection_type]['failed'] += 1
            self.stats.by_template[template]['failed'] += 1
            self.stats.failed_contracts.append({
                'file': str(sol_file),
                'type': injection_type,
                'template': template,
                'error': result.error_short or 'Unknown error'
            })
        
        self.stats.by_type[injection_type]['total'] += 1
        self.stats.by_template[template]['total'] += 1
    
    def _get_injection_info(self, sol_file: str) -> tuple[str, str, Optional[dict]]:
        """
        Get injection information from metadata JSON or filename.
        
        Args:
            sol_file: Path to the .sol file
            
        Returns:
            Tuple of (injection_type, template_name, metadata_dict)
        """
        json_file = sol_file.replace('.sol', '.json')
        
        if os.path.exists(json_file):
            return self._parse_metadata_file(json_file)
        
        # Fallback to filename parsing
        return self._parse_filename(sol_file)
    
    def _parse_metadata_file(self, json_file: str) -> tuple[str, str, Optional[dict]]:
        """Parse injection metadata from JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            injection_mode = metadata.get('injection_mode', 'unknown')
            vulnerability_type = metadata.get('vulnerability_type', 'unknown')
            template_name = metadata.get('template_name', 'unknown')
            
            if injection_mode == 'point':
                return 'point', vulnerability_type, metadata
            elif injection_mode == 'coupled':
                return 'coupled', template_name, metadata
            else:
                return injection_mode, vulnerability_type, metadata
                
        except Exception:
            # Fall back to filename parsing
            return self._parse_filename(json_file.replace('.json', '.sol'))
    
    def _parse_filename(self, sol_file: str) -> tuple[str, str, None]:
        """Parse injection type from filename pattern."""
        filename = os.path.basename(sol_file)
        
        if '_point_' in filename:
            return 'point', 'reentrancy', None
        elif '_coupled_' in filename:
            parts = filename.split('_coupled_')
            if len(parts) > 1:
                template_part = parts[1].replace('.sol', '')
                template = template_part.split('_')[1] if '_' in template_part else template_part
                return 'coupled', template, None
            return 'coupled', 'unknown', None
        
        return 'unknown', 'unknown', None
    
    def save_report(self, output_path: str | Path) -> None:
        """
        Save validation report to JSON file.
        
        Args:
            output_path: Path for the output JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(self.stats.to_dict(), f, indent=2)
    
    def print_summary(self) -> None:
        """Print a summary of validation results."""
        print("=" * 80)
        print("COMPILATION TEST RESULTS")
        print("=" * 80)
        print(f"\nTotal contracts tested: {self.stats.total}")
        print(f"✅ Successful: {self.stats.success} ({self.stats.success_rate:.2f}%)")
        print(f"❌ Failed: {self.stats.failed} ({100 - self.stats.success_rate:.2f}%)")
        
        print("\n" + "-" * 80)
        print("STATISTICS BY INJECTION TYPE")
        print("-" * 80)
        
        for inj_type in sorted(self.stats.by_type.keys()):
            type_stats = self.stats.by_type[inj_type]
            rate = (type_stats['success'] / type_stats['total'] * 100) if type_stats['total'] > 0 else 0
            print(f"\n{inj_type.upper()}:")
            print(f"  Total: {type_stats['total']}")
            print(f"  Success: {type_stats['success']} ({rate:.2f}%)")
            print(f"  Failed: {type_stats['failed']} ({100-rate:.2f}%)")
        
        print("\n" + "-" * 80)
        print("STATISTICS BY TEMPLATE")
        print("-" * 80)
        
        # Sort by failure rate
        template_items = sorted(
            self.stats.by_template.items(),
            key=lambda x: (x[1]['failed'] / x[1]['total'] if x[1]['total'] > 0 else 0),
            reverse=True
        )
        
        for template, template_stats in template_items:
            rate = (template_stats['success'] / template_stats['total'] * 100) if template_stats['total'] > 0 else 0
            print(f"\n{template}:")
            print(f"  Total: {template_stats['total']}")
            print(f"  Success: {template_stats['success']} ({rate:.2f}%)")
            print(f"  Failed: {template_stats['failed']} ({100-rate:.2f}%)")
