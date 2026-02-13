#!/usr/bin/env python3
"""
Verify the prepared instruction fine-tuning data.

Quick checks for data quality and format compliance.

Usage:
    python llm-based/verify_data.py
    python llm-based/verify_data.py --data-dir llm-based/data
"""
import json
import argparse
from pathlib import Path
from collections import Counter


def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def verify_sharegpt_format(conversations):
    """Verify ShareGPT format compliance."""
    issues = []
    
    for i, sample in enumerate(conversations):
        # Check if 'conversations' key exists
        if 'conversations' not in sample:
            issues.append(f"Sample {i}: Missing 'conversations' key")
            continue
        
        convs = sample['conversations']
        
        # Check if it's a list
        if not isinstance(convs, list):
            issues.append(f"Sample {i}: 'conversations' is not a list")
            continue
        
        # Check if there are at least 2 turns (human + gpt)
        if len(convs) < 2:
            issues.append(f"Sample {i}: Less than 2 conversation turns")
            continue
        
        # Check each turn
        for j, turn in enumerate(convs):
            if 'from' not in turn:
                issues.append(f"Sample {i}, turn {j}: Missing 'from' field")
            elif turn['from'] not in ['human', 'gpt', 'system']:
                issues.append(f"Sample {i}, turn {j}: Invalid 'from' value: {turn['from']}")
            
            if 'value' not in turn:
                issues.append(f"Sample {i}, turn {j}: Missing 'value' field")
            elif not isinstance(turn['value'], str):
                issues.append(f"Sample {i}, turn {j}: 'value' is not a string")
    
    return issues


def analyze_dataset(filepath):
    """Analyze a single dataset file."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {filepath.name}")
    print(f"{'='*70}")
    
    # Load data
    data = load_json(filepath)
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Total samples: {len(data)}")
    
    # Check format
    issues = verify_sharegpt_format(data)
    if issues:
        print(f"\n⚠️  Format Issues ({len(issues)} found):")
        for issue in issues[:5]:  # Show first 5
            print(f"    - {issue}")
        if len(issues) > 5:
            print(f"    ... and {len(issues) - 5} more")
    else:
        print(f"\n✅ Format: Valid ShareGPT format")
    
    # Content analysis
    vuln_count = 0
    clean_count = 0
    response_lengths = []
    question_lengths = []
    
    for sample in data:
        if 'conversations' not in sample:
            continue
        
        convs = sample['conversations']
        
        # Get the assistant response
        gpt_response = None
        human_question = None
        
        for turn in convs:
            if turn.get('from') == 'gpt':
                gpt_response = turn.get('value', '')
            if turn.get('from') == 'human':
                human_question = turn.get('value', '')
        
        if gpt_response:
            response_lengths.append(len(gpt_response))
            
            # Count vulnerable vs clean
            if 'VULNERABLE' in gpt_response or 'vulnerable' in gpt_response.lower():
                vuln_count += 1
            else:
                clean_count += 1
        
        if human_question:
            question_lengths.append(len(human_question))
    
    # Distribution
    print(f"\n📈 Distribution:")
    print(f"  Vulnerable: {vuln_count} ({vuln_count/len(data)*100:.1f}%)")
    print(f"  Clean: {clean_count} ({clean_count/len(data)*100:.1f}%)")
    
    # Length statistics
    if response_lengths:
        avg_response = sum(response_lengths) / len(response_lengths)
        max_response = max(response_lengths)
        min_response = min(response_lengths)
        
        print(f"\n📝 Response Lengths:")
        print(f"  Average: {avg_response:.0f} characters")
        print(f"  Min: {min_response} characters")
        print(f"  Max: {max_response} characters")
    
    if question_lengths:
        avg_question = sum(question_lengths) / len(question_lengths)
        max_question = max(question_lengths)
        
        print(f"\n❓ Question Lengths:")
        print(f"  Average: {avg_question:.0f} characters")
        print(f"  Max: {max_question} characters")
        
        # Warn if questions are too long
        if max_question > 10000:
            print(f"  ⚠️  Warning: Some questions are very long (>{max_question} chars)")
    
    # Sample a few examples
    print(f"\n📋 Sample Conversations:")
    for i in range(min(2, len(data))):
        sample = data[i]
        if 'conversations' in sample:
            convs = sample['conversations']
            print(f"\n  Example {i+1}:")
            for turn in convs:
                role = turn.get('from', 'unknown')
                content = turn.get('value', '')
                preview = content[:100] + '...' if len(content) > 100 else content
                print(f"    {role}: {preview}")


def main():
    parser = argparse.ArgumentParser(description="Verify prepared instruction data")
    parser.add_argument(
        '--data-dir',
        type=str,
        default='llm-based/data',
        help='Directory containing the data files'
    )
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    
    print("="*70)
    print("INSTRUCTION DATA VERIFICATION")
    print("="*70)
    
    # Check if directory exists
    if not data_dir.exists():
        print(f"\n❌ Error: Data directory not found: {data_dir}")
        print("Run the data preparation script first:")
        print("  python llm-based/prepare_instruction_data.py")
        return 1
    
    # Find all JSON files
    json_files = sorted(data_dir.glob('sc_*.json'))
    
    if not json_files:
        print(f"\n❌ Error: No data files found in {data_dir}")
        print("Run the data preparation script first:")
        print("  python llm-based/prepare_instruction_data.py")
        return 1
    
    print(f"\nFound {len(json_files)} data files:")
    for f in json_files:
        print(f"  - {f.name}")
    
    # Analyze each file
    all_valid = True
    for filepath in json_files:
        try:
            analyze_dataset(filepath)
        except Exception as e:
            print(f"\n❌ Error analyzing {filepath.name}: {e}")
            all_valid = False
    
    # Summary
    print(f"\n{'='*70}")
    if all_valid:
        print("✅ ALL DATASETS VERIFIED SUCCESSFULLY!")
        print(f"{'='*70}")
        print("\nYou can now proceed with fine-tuning:")
        print("  cd llm-based")
        print("  bash quickstart_instruct.sh")
    else:
        print("⚠️  SOME ISSUES FOUND")
        print(f"{'='*70}")
        print("\nPlease fix the issues above before proceeding.")
    
    return 0 if all_valid else 1


if __name__ == '__main__':
    exit(main())
