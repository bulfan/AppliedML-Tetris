#!/usr/bin/env python3
"""
Clean Comments and Emojis Script
Removes:
- Lines that are only comments (start with- Inline comments (everything after- Emojis from text
- Optionally preserves shebang lines and important comments
"""
import re
import os
import argparse
import sys
from pathlib import Path
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"    "\U0001F300-\U0001F5FF"    "\U0001F680-\U0001F6FF"    "\U0001F1E0-\U0001F1FF"    "\U00002702-\U000027B0"    "\U000024C2-\U0001F251"    "\U0001F900-\U0001F9FF"    "\U0001F018-\U0001F270"    "\U0001F300-\U0001F5FF"    "]+", 
    flags=re.UNICODE
)
def remove_emojis(text):
    """Remove emojis from text."""
    return EMOJI_PATTERN.sub('', text)
def clean_line(line, preserve_important=True):
    """
    Clean a single line by removing comments and emojis.
    Args:
        line (str): The line to clean
        preserve_important (bool): Whether to preserve shebang and encoding comments
    Returns:
        str: Cleaned line, or None if line should be removed
    """
    original_line = line.rstrip()
    if preserve_important and line.startswith('#!'):
        return remove_emojis(line)
    if preserve_important and ('coding:' in line or 'encoding:' in line) and line.strip().startswith('#'):
        return remove_emojis(line)
    stripped = line.strip()
    if stripped.startswith('#'):
        return None
    if '#' in line:
        in_single_quote = False
        in_double_quote = False
        in_triple_single = False
        in_triple_double = False
        i = 0
        while i < len(line):
            char = line[i]
            if i <= len(line) - 3:
                three_chars = line[i:i+3]
                if three_chars == '"""' and not in_single_quote:
                    in_triple_double = not in_triple_double
                    i += 3
                    continue
                elif three_chars == "'''" and not in_double_quote:
                    in_triple_single = not in_triple_single
                    i += 3
                    continue
            if in_triple_single or in_triple_double:
                i += 1
                continue
            if char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            elif char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '#' and not (in_single_quote or in_double_quote):
                line = line[:i].rstrip()
                break
            i += 1
    line = remove_emojis(line)
    if not line.strip():
        return None
    return line
def clean_file(file_path, preserve_important=True, backup=True):
    """
    Clean a single file by removing comments and emojis.
    Args:
        file_path (str): Path to the file to clean
        preserve_important (bool): Whether to preserve important comments
        backup (bool): Whether to create a backup before cleaning
    Returns:
        tuple: (success, message)
    """
    try:
        file_path = Path(file_path)
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if backup:
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"Created backup: {backup_path}")
        cleaned_lines = []
        removed_count = 0
        for line_num, line in enumerate(lines, 1):
            cleaned_line = clean_line(line, preserve_important)
            if cleaned_line is None:
                removed_count += 1
            else:
                cleaned_lines.append(cleaned_line)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        return True, f"Cleaned {file_path}: removed {removed_count} comment lines"
    except Exception as e:
        return False, f"Error cleaning {file_path}: {str(e)}"
def should_process_file(file_path, extensions):
    """Check if file should be processed based on extension."""
    if not extensions:
        return True
    return file_path.suffix.lower() in extensions
def clean_directory(directory, extensions=None, preserve_important=True, backup=True, recursive=True):
    """
    Clean all files in a directory.
    Args:
        directory (str): Directory path
        extensions (list): List of extensions to process (e.g., ['.py', '.txt'])
        preserve_important (bool): Whether to preserve important comments
        backup (bool): Whether to create backups
        recursive (bool): Whether to process subdirectories
    Returns:
        tuple: (total_files, successful_files, errors)
    """
    directory = Path(directory)
    total_files = 0
    successful_files = 0
    errors = []
    pattern = "**/*" if recursive else "*"
    for file_path in directory.glob(pattern):
        if file_path.is_file() and should_process_file(file_path, extensions):
            total_files += 1
            success, message = clean_file(file_path, preserve_important, backup)
            if success:
                successful_files += 1
                print(f" {message}")
            else:
                errors.append(message)
                print(f" {message}")
    return total_files, successful_files, errors
def main():
    parser = argparse.ArgumentParser(
        description="Remove comments (starting with #) and emojis from files",
        epilog="""
Examples:
  python clean_comments_emojis.py file.py
  python clean_comments_emojis.py . --ext .py
  python clean_comments_emojis.py . --recursive --no-backup
  python clean_comments_emojis.py . --ext .py .txt .md --preserve-important
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('path', help='File or directory path to clean')
    parser.add_argument('--ext', '--extensions', nargs='+', 
                       help='File extensions to process (e.g., .py .txt .md)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Process subdirectories recursively')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not create backup files')
    parser.add_argument('--preserve-important', action='store_true', default=True,
                       help='Preserve shebang and encoding comments (default: True)')
    parser.add_argument('--remove-all-comments', action='store_true',
                       help='Remove ALL comments including shebang and encoding')
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        print(f" Error: Path '{path}' does not exist")
        sys.exit(1)
    preserve_important = args.preserve_important and not args.remove_all_comments
    backup = not args.no_backup
    print(f" Cleaning comments and emojis from: {path}")
    print(f" Settings:")
    print(f"   - Preserve important comments: {preserve_important}")
    print(f"   - Create backups: {backup}")
    print(f"   - Extensions: {args.ext or 'All files'}")
    print(f"   - Recursive: {args.recursive}")
    print()
    if path.is_file():
        if args.ext and not should_process_file(path, args.ext):
            print(f"⏭  Skipping {path} (extension not in filter)")
            return
        success, message = clean_file(path, preserve_important, backup)
        if success:
            print(f" {message}")
        else:
            print(f" {message}")
            sys.exit(1)
    else:
        total, successful, errors = clean_directory(
            path, args.ext, preserve_important, backup, args.recursive
        )
        print(f"\n Summary:")
        print(f"   Total files processed: {total}")
        print(f"   Successfully cleaned: {successful}")
        print(f"   Errors: {len(errors)}")
        if errors:
            print(f"\n Errors:")
            for error in errors:
                print(f"   {error}")
            sys.exit(1)
        if total == 0:
            print("  No files found to process")
        else:
            print(" All files cleaned successfully!")
if __name__ == "__main__":
    main() 