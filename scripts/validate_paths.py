from pathlib import Path
import sys


def validate_structure():
    """Validate the new data folder structure."""
    
    print("=" * 70)
    print("DATA FOLDER STRUCTURE VALIDATION")
    print("=" * 70)
    
    # Define expected structure
    expected_paths = {
        'Synthetic Data': [
            'data/synthetic/sc-source/clean',
            'data/synthetic/sc-source/vulnerable',
            'data/synthetic/ast_dataset',
            'data/synthetic/function_level_dataset',
        ],
        'Real-world Data': [
            'data/realworld/sc-source',
            'data/realworld/ast_dataset',
            'data/realworld/function_level_dataset',
        ],
    }
    
    all_valid = True
    
    for category, paths in expected_paths.items():
        print(f"\n{category}:")
        print("-" * 70)
        
        for path_str in paths:
            path = Path(path_str)
            exists = path.exists()
            is_dir = path.is_dir() if exists else False
            
            # Count files if directory exists
            file_count = "N/A"
            if is_dir:
                try:
                    files = list(path.rglob('*.*'))
                    file_count = len(files)
                except:
                    file_count = "Error"
            
            # Status symbol
            status = "✓" if exists else "✗"
            
            # Print result
            if exists:
                print(f"  {status} {path_str:<50} ({file_count} files)")
            else:
                print(f"  {status} {path_str:<50} MISSING")
                all_valid = False
    
    print("\n" + "=" * 70)
    
    if all_valid:
        print("✓ All expected paths exist!")
        print("\nNext steps:")
        print("  1. If datasets are empty, rebuild them:")
        print("     python dataset_builder.py --input data/synthetic/sc-source/vulnerable --output data/synthetic/ast_dataset")
        print("  2. Test training:")
        print("     python main.py train binary --epochs 1")
        return 0
    else:
        print("✗ Some paths are missing!")
        print("\nTo fix:")
        print("  1. Run the reorganization script:")
        print("     bash reorganize_data.sh")
        print("  2. Rebuild datasets if needed")
        return 1


if __name__ == '__main__':
    sys.exit(validate_structure())
