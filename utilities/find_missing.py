"""
Find Missing Files

Finds missing files in a numbered sequence (e.g., image001.png, image002.png, image005.png
would report image003.png and image004.png as missing).
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional


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


def find_missing_files(folder_path: Path) -> list[str]:
    """
    Find missing files in a numbered sequence.
    
    Args:
        folder_path: Path to folder containing numbered files
        
    Returns:
        List of missing filenames
    """
    try:
    all_files = sorted(os.listdir(folder_path))
    except OSError as e:
        logging.error(f"Error reading directory {folder_path}: {e}")
        return []
    
    missing_files: list[str] = []

    for i in range(len(all_files) - 1):
        current_file = all_files[i]
        next_file = all_files[i + 1]

        current_file_name, current_file_ext = os.path.splitext(current_file)
        next_file_name, next_file_ext = os.path.splitext(next_file)

        if current_file_ext != next_file_ext:
            continue

        try:
        current_file_number = int(''.join(filter(str.isdigit, current_file_name)))
        next_file_number = int(''.join(filter(str.isdigit, next_file_name)))
        except ValueError:
            # Skip files that don't have extractable numbers
            continue

        if next_file_number - current_file_number > 1:
            missing_range = list(range(current_file_number + 1, next_file_number))
            for number in missing_range:
                missing_file = current_file_name.rstrip('0123456789') + str(number) + current_file_ext
                missing_files.append(missing_file)

    return missing_files


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Find missing files in a numbered sequence."
    )
    parser.add_argument(
        "-f", "--folder",
        type=str,
        required=True,
        help="Folder path containing numbered files"
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
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    folder_path = Path(args.folder).expanduser().resolve()
    
    if not folder_path.exists() or not folder_path.is_dir():
        logging.error(f"'{folder_path}' is not a valid directory.")
        sys.exit(1)

        missing_files = find_missing_files(folder_path)

        if missing_files:
        output_text = "Missing files:\n" + "\n".join(missing_files) + "\n"
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.write_text(output_text, encoding="utf-8")
            logging.info(f"Missing files list saved to: {output_path}")
        else:
            print(output_text, end="")
        logging.info(f"Found {len(missing_files)} missing file(s).")
    else:
        message = "No files are missing."
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.write_text(message + "\n", encoding="utf-8")
            logging.info(f"Result saved to: {output_path}")
        else:
            print(message)
        logging.info(message)


if __name__ == "__main__":
    main()
