import argparse
from pathlib import Path
import re


def contains_code_block(content: str) -> bool:
    return bool(re.search(r"```.*?```", content, re.DOTALL))


def replace_latex_syntax_in_file(file_path: str, skip_code_blocks: bool = False):
    """Replace legacy LaTeX delimiters in one Markdown file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    if skip_code_blocks and contains_code_block(content):
        print(f"Skipping file with code block: {file_path}")
        return False

    updated_content = re.sub(r'\\\[', '$$', content)
    updated_content = re.sub(r'\\\]', '$$', updated_content)
    updated_content = re.sub(r'\\\(', '$$', updated_content)
    updated_content = re.sub(r'\\\)', '$$', updated_content)

    if updated_content == content:
        return False

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    return True


def process_markdown_files_in_folder(folder_path: str, skip_code_blocks: bool = False):
    """Process Markdown files recursively below ``folder_path``."""
    for file_path in sorted(Path(folder_path).rglob('*.md')):
        print(f'Processing file: {file_path}')
        replace_latex_syntax_in_file(file_path, skip_code_blocks=skip_code_blocks)
        print(f'Finished processing file: {file_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace LaTeX delimiters in markdown files")
    parser.add_argument("--path", default="./_posts", help="Target folder")
    parser.add_argument("--skip-code-blocks", action="store_true", help="Skip files containing fenced code blocks")
    args = parser.parse_args()
    process_markdown_files_in_folder(args.path, skip_code_blocks=args.skip_code_blocks)
