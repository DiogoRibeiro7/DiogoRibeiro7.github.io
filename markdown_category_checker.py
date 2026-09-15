import argparse
import json
from pathlib import Path
from typing import List

import frontmatter


def read_markdown_files_from_folder(folder_path: str) -> List[str]:
    root = Path(folder_path)
    return [path.relative_to(root).as_posix() for path in sorted(root.rglob('*.md'))]


def check_categories(frontmatter_data: dict) -> bool:
    if 'categories' in frontmatter_data and isinstance(frontmatter_data['categories'], list):
        return len(frontmatter_data['categories']) > 1
    return False


def process_markdown_files(folder_path: str, output_file: str, output_format: str = "text"):
    root = Path(folder_path)
    markdown_files = read_markdown_files_from_folder(folder_path)
    files_with_multiple_categories = []

    for md_file in markdown_files:
        post = frontmatter.load(root / md_file)
        frontmatter_data = dict(post.metadata)
        if check_categories(frontmatter_data):
            files_with_multiple_categories.append({
                "file": md_file,
                "categories": frontmatter_data.get("categories", []),
            })

    with open(output_file, 'w', encoding='utf-8') as output:
        if output_format == "json":
            json.dump(files_with_multiple_categories, output, indent=2)
            output.write("\n")
        else:
            for result in files_with_multiple_categories:
                output.write(f'{result["file"]}\n')

    return files_with_multiple_categories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check categories in markdown files")
    parser.add_argument("--path", default="./_posts", help="Target folder")
    parser.add_argument("--output", default="files_with_multiple_categories.txt", help="Output file")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()
    process_markdown_files(args.path, args.output, args.format)
    print(f'Processing complete. Files with multiple categories saved to {args.output}')
