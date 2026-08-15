import os
import re
import yaml
import argparse
import json
from typing import List

def read_markdown_files_from_folder(folder_path: str) -> List[str]:
    # List all markdown files in the given folder
    return [f for f in os.listdir(folder_path) if f.endswith('.md')]

def extract_frontmatter(file_content: str) -> dict:
    # Extract the YAML frontmatter from the markdown file using regex
    frontmatter_match = re.match(r'---\n(.*?)\n---', file_content, re.DOTALL)
    if frontmatter_match:
        frontmatter_str = frontmatter_match.group(1)
        try:
            return yaml.safe_load(frontmatter_str)
        except yaml.YAMLError:
            return {}
    return {}

def check_categories(frontmatter: dict) -> bool:
    # Check if 'categories' key exists and contains more than one element
    if 'categories' in frontmatter and isinstance(frontmatter['categories'], list):
        return len(frontmatter['categories']) > 1
    return False

def process_markdown_files(folder_path: str, output_file: str, output_format: str = "text"):
    markdown_files = read_markdown_files_from_folder(folder_path)
    files_with_multiple_categories = []

    for md_file in markdown_files:
        with open(os.path.join(folder_path, md_file), 'r', encoding='utf-8') as file:
            content = file.read()
            frontmatter = extract_frontmatter(content)
            if check_categories(frontmatter):
                files_with_multiple_categories.append({
                    "file": md_file,
                    "categories": frontmatter.get("categories", []),
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
