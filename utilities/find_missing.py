"""
Find Missing Files

Verifies that for a given directory tree, files increment from 0-N equally in each subfolder.
Reports missing files and range mismatches across subfolders.
"""
import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )


def extract_file_number(filename: str) -> Optional[int]:
    """
    Extract the last numeric sequence from filename (before extension).
    This handles filenames like "face_digital_40x_1910.webp" -> 1910
    
    Args:
        filename: Filename to extract number from
        
    Returns:
        Extracted number or None if no number found
    """
    file_name, _ = os.path.splitext(filename)
    
    # Find the last sequence of digits at the end of the filename
    # This handles cases like "file_40x_1910" -> 1910, not 401910
    match = re.search(r'(\d+)(?!.*\d)', file_name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def get_expected_file_count(subfolder_data: Dict[Path, List[int]]) -> int:
    """
    Determine expected file count from all subfolders.
    All subfolders should have the same number of files.
    
    Args:
        subfolder_data: Dictionary mapping subfolder paths to lists of file numbers
        
    Returns:
        Expected file count (should be the same across all subfolders)
    """
    counts = [len(numbers) for numbers in subfolder_data.values() if numbers]
    if not counts:
        return -1
    
    # All folders should have the same count, so return the most common count
    # (or max if they differ, to catch mismatches)
    return max(counts)


def find_missing_in_subfolder(subfolder_path: Path, expected_count: int, file_template: Optional[str] = None) -> Tuple[List[str], int, int, List[int]]:
    """
    Find missing files in a specific subfolder for range 0 to (expected_count - 1).
    
    Args:
        subfolder_path: Path to subfolder to check
        expected_count: Expected number of files (files should be numbered 0 to expected_count-1)
        file_template: Optional template for missing filename (prefix + number + extension)
        
    Returns:
        Tuple of (list of missing filenames, actual file count, actual maximum number found, list of duplicate numbers)
    """
    try:
        all_files = sorted(os.listdir(subfolder_path))
    except OSError as e:
        logging.error(f"Error reading directory {subfolder_path}: {e}")
        return [], 0, -1, []
    
    # Extract numbers from all files
    file_numbers: Dict[int, str] = {}  # number -> filename
    duplicate_numbers: List[int] = []  # Track duplicate numbers
    for filename in all_files:
        number = extract_file_number(filename)
        if number is not None:
            if number in file_numbers:
                # Duplicate number found
                if number not in duplicate_numbers:
                    duplicate_numbers.append(number)
            else:
                file_numbers[number] = filename
    
    if not file_numbers:
        return [], 0, -1, []
    
    actual_count = len(file_numbers)
    actual_max = max(file_numbers.keys())
    missing_files: List[str] = []
    
    # Determine file template from existing files if not provided
    if file_template is None and file_numbers:
        # Use the first file as template
        sample_file = list(file_numbers.values())[0]
        file_name, file_ext = os.path.splitext(sample_file)
        # Remove digits to get prefix
        prefix = file_name.rstrip('0123456789')
        file_template = prefix + "{:d}" + file_ext
    
    # Check for missing numbers from 0 to (expected_count - 1)
    for num in range(expected_count):
        if num not in file_numbers:
            if file_template:
                missing_filename = file_template.format(num)
            else:
                missing_filename = f"file{num:03d}"
            missing_files.append(missing_filename)
    
    return missing_files, actual_count, actual_max, duplicate_numbers


def find_missing_files(root_path: Path) -> Dict[Path, Dict[str, Any]]:
    """
    Traverse directory tree and verify that each subfolder has files numbered 0-N with no gaps.
    Also verifies that all subfolders have the same maximum number N.
    
    Args:
        root_path: Root directory path to traverse
        
    Returns:
        Dictionary mapping subfolder paths to their analysis results:
        {
            'missing_files': List[str],
            'actual_max': int,
            'expected_max': int,
            'has_range_mismatch': bool
        }
    """
    results: Dict[Path, Dict[str, Any]] = {}
    
    # Step 1: Collect all subfolders and their file numbers
    subfolder_data: Dict[Path, List[int]] = {}
    subfolder_file_counts: Dict[Path, int] = {}  # Track actual file counts (including duplicates)
    
    for root, dirs, files in os.walk(root_path):
        subfolder = Path(root)
        
        # Extract numbers from files in this subfolder
        file_numbers: List[int] = []
        actual_file_count = 0
        for filename in files:
            number = extract_file_number(filename)
            if number is not None:
                file_numbers.append(number)
                actual_file_count += 1
        
        if file_numbers:
            subfolder_data[subfolder] = file_numbers
            subfolder_file_counts[subfolder] = actual_file_count
    
    if not subfolder_data:
        logging.warning(f"No numbered files found in directory tree: {root_path}")
        return results
    
    # Step 2: Find expected file count (all folders should have the same count)
    expected_count = get_expected_file_count(subfolder_data)
    
    if expected_count < 0:
        logging.warning("No valid file numbers found in directory tree")
        return results
    
    # Step 3: Check each subfolder for missing files and range mismatches
    for subfolder, file_numbers in subfolder_data.items():
        # Use actual file count (including duplicates), not just unique numbers
        actual_count = subfolder_file_counts.get(subfolder, len(file_numbers))
        unique_numbers = len(set(file_numbers))  # Count of unique numbers
        actual_max = max(file_numbers) if file_numbers else -1
        
        # Determine file template from existing files
        sample_file = None
        for filename in sorted(os.listdir(subfolder)):
            if extract_file_number(filename) is not None:
                sample_file = filename
                break
        
        file_template = None
        if sample_file:
            file_name, file_ext = os.path.splitext(sample_file)
            prefix = file_name.rstrip('0123456789')
            file_template = prefix + "{:d}" + file_ext
        
        # Find missing files (check for 0 to expected_count-1)
        missing_files, _, _, duplicate_numbers = find_missing_in_subfolder(subfolder, expected_count, file_template)
        
        # Range mismatch occurs when file count differs from expected
        has_range_mismatch = (actual_count != expected_count)
        has_duplicates = len(duplicate_numbers) > 0
        
        results[subfolder] = {
            'missing_files': missing_files,
            'actual_count': actual_count,
            'expected_count': expected_count,
            'actual_max': actual_max,
            'has_range_mismatch': has_range_mismatch,
            'duplicate_numbers': duplicate_numbers,
            'has_duplicates': has_duplicates
        }
    
    return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Find missing files in a numbered sequence."
    )
    parser.add_argument(
        "-f", "--folder",
        type=str,
        required=True,
        help="Root directory path to traverse (checks all subfolders)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output file path to write missing files list (default: print to stdout)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    start_time = time.time()
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    root_path = Path(args.folder).expanduser().resolve()
    
    if not root_path.exists() or not root_path.is_dir():
        logging.error(f"'{root_path}' is not a valid directory.")
        sys.exit(1)

    results = find_missing_files(root_path)
    elapsed_time = time.time() - start_time

    if not results:
        message = "No numbered files found in directory tree."
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.write_text(message + "\n", encoding="utf-8")
            logging.info(f"Result saved to: {output_path}")
        else:
            print(message)
        logging.info(message)
        return

    # Build output report - only show subfolders with issues
    output_lines: List[str] = []
    has_issues = False
    
    # Sort subfolders for consistent output
    sorted_subfolders = sorted(results.keys(), key=lambda p: str(p))
    
    for subfolder in sorted_subfolders:
        data = results[subfolder]
        missing_files = data['missing_files']
        actual_count = data['actual_count']
        expected_count = data['expected_count']
        actual_max = data['actual_max']
        has_range_mismatch = data['has_range_mismatch']
        
        # Only include subfolders with issues
        has_duplicates = data.get('has_duplicates', False)
        if not missing_files and not has_range_mismatch and not has_duplicates:
            continue

        has_issues = True
        
        # Relative path for cleaner output
        rel_path = subfolder.relative_to(root_path) if subfolder != root_path else Path(".")
        subfolder_name = str(rel_path) if str(rel_path) != "." else root_path.name
        
        if missing_files or has_range_mismatch or has_duplicates:
            issue_parts = []
            if has_duplicates:
                dup_numbers = data.get('duplicate_numbers', [])
                if len(dup_numbers) <= 10:
                    issue_parts.append(f"duplicate numbers: {', '.join(map(str, sorted(dup_numbers)))}")
                else:
                    issue_parts.append(f"duplicate numbers: {len(dup_numbers)} duplicates (e.g., {', '.join(map(str, sorted(dup_numbers)[:5]))}...)")
            if missing_files:
                missing_count = len(missing_files)
                # Extract numbers from missing files to show ranges
                missing_numbers = []
                for f in missing_files:
                    num = extract_file_number(f)
                    if num is not None:
                        missing_numbers.append(num)
                
                if missing_numbers:
                    missing_numbers.sort()
                    if len(missing_numbers) <= 20:
                        # Show all missing numbers if 20 or fewer
                        issue_parts.append(f"missing {missing_count} files: {', '.join(map(str, missing_numbers))}")
                    else:
                        # Show range for many missing files
                        issue_parts.append(f"missing {missing_count} files: {min(missing_numbers)}-{max(missing_numbers)}")
                else:
                    issue_parts.append(f"missing {missing_count} files")
            if has_range_mismatch:
                issue_parts.append(f"count mismatch ({actual_count} vs {expected_count})")
            issue_text = " | ".join(issue_parts)
            output_lines.append(f"{subfolder_name}: {issue_text}")
    
    output_text = "\n".join(output_lines)
    
    if has_issues:
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.write_text(output_text, encoding="utf-8")
            logging.info(f"Report saved to: {output_path}")
        else:
            print(output_text, end="")
        logging.info("Issues found in directory tree (see above)")
    else:
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.write_text(output_text, encoding="utf-8")
            logging.info(f"Report saved to: {output_path}")
        else:
            print(output_text, end="")
        logging.info("All subfolders have complete 0-N sequences with matching ranges.")
    
    # Summary
    total_folders = len(results)
    folders_with_issues = sum(1 for data in results.values() if data.get('missing_files') or data.get('has_range_mismatch') or data.get('has_duplicates', False))
    folders_with_missing = sum(1 for data in results.values() if data.get('missing_files'))
    folders_with_mismatch = sum(1 for data in results.values() if data.get('has_range_mismatch'))
    folders_with_duplicates = sum(1 for data in results.values() if data.get('has_duplicates', False))
    total_missing_files = sum(len(data.get('missing_files', [])) for data in results.values())
    
    summary_lines = [
        "",
        "Summary:",
        f"  Folders checked: {total_folders}",
        f"  Folders with issues: {folders_with_issues}",
        f"  - Missing files: {folders_with_missing}",
        f"  - Count mismatches: {folders_with_mismatch}",
        f"  - Duplicate numbers: {folders_with_duplicates}",
        f"  Total missing files: {total_missing_files}",
        f"  Time elapsed: {elapsed_time:.2f} seconds"
    ]
    
    summary_text = "\n".join(summary_lines)
    if args.output:
        # Append summary to output file
        output_path = Path(args.output).expanduser().resolve()
        with open(output_path, 'a', encoding="utf-8") as f:
            f.write(summary_text + "\n")
    else:
        print(summary_text)
    logging.info(f"Checked {total_folders} folders in {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()
