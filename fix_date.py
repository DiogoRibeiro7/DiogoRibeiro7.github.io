import argparse
import os
from pathlib import Path
import re

import frontmatter


DATE_LINE_PATTERN = re.compile(r"^date:\s*.*$", re.MULTILINE)


def extract_date_from_filename(filename):
    # Assuming the filename format is 'YYYY-MM-DD-some-title.md'
    match = re.match(r'(\d{4}-\d{2}-\d{2})-', filename)
    if match:
        return match.group(1)
    return None


def process_markdown_file(filepath, dry_run=False):
    path = Path(filepath)
    filename = os.path.basename(filepath)
    file_date = extract_date_from_filename(filename)

    if not file_date:
        print(f"Could not extract date from filename: {filename}")
        return False

    post = frontmatter.load(path)
    frontmatter_date = post.get('date')

    if frontmatter_date is not None and str(frontmatter_date) == file_date:
        return False

    if frontmatter_date is None:
        print(f"Adding date to {filename}: {file_date}")
    else:
        print(f"Updating date in {filename}: {frontmatter_date} -> {file_date}")

    if dry_run:
        print(f"Dry run: would update {filename}")
        return True

    # Change only the date field. Re-serializing the entire front matter block is
    # unnecessary for this utility and can alter unrelated YAML values.
    text = path.read_text(encoding='utf-8')
    replacement = f"date: '{file_date}'"

    if frontmatter_date is None:
        if not text.startswith('---\n'):
            raise ValueError(f"Missing YAML front matter: {filepath}")
        updated_text = '---\n' + replacement + '\n' + text[4:]
    else:
        updated_text, replacements = DATE_LINE_PATTERN.subn(replacement, text, count=1)
        if replacements != 1:
            raise ValueError(f"Could not locate date field in front matter: {filepath}")

    path.write_text(updated_text, encoding='utf-8')
    return True


def process_markdown_files_in_directory(directory, dry_run=False):
    # Subject folders under _posts are part of the supported repository layout.
    for filepath in sorted(Path(directory).rglob('*.md')):
        process_markdown_file(filepath, dry_run=dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix dates in markdown front matter")
    parser.add_argument("--path", default="./_posts", help="Target folder")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing files")
    args = parser.parse_args()
    process_markdown_files_in_directory(args.path, dry_run=args.dry_run)
