import argparse
import sys
from pathlib import Path

from testing import InjectedContractViewer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="View injected vulnerabilities in smart contracts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python view_injected.py data/synthetic/sc-source/vulnerable/contract_point_0.json
  python view_injected.py data/synthetic/sc-source/vulnerable/contract_coupled_0.json --only
"""
    )
    parser.add_argument("metadata_file", help="Path to the metadata JSON file")
    parser.add_argument("--only", action="store_true", help="Show only injected code without context")
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()
    
    metadata_path = Path(args.metadata_file)
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return 1
    
    try:
        viewer = InjectedContractViewer(metadata_path)
        viewer.load()
        
        if args.only:
            viewer.display_code_only()
        else:
            viewer.display_with_context()
        
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
