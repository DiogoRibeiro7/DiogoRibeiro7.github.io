import os
import re
import yaml
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

def process_markdown_files(folder_path: str, output_txt_file: str):
    markdown_files = read_markdown_files_from_folder(folder_path)
    files_with_multiple_categories = []

    for md_file in markdown_files:
        with open(os.path.join(folder_path, md_file), 'r', encoding='utf-8') as file:
            content = file.read()
            frontmatter = extract_frontmatter(content)
            if check_categories(frontmatter):
                files_with_multiple_categories.append(md_file)

    # Write filenames to output text file
    with open(output_txt_file, 'w', encoding='utf-8') as output_file:
        for filename in files_with_multiple_categories:
            output_file.write(f'{filename}\n')


folder_path = './_posts'  # Change this to your folder path
output_txt_file = 'files_with_multiple_categories.txt'
process_markdown_files(folder_path, output_txt_file)
print(f'Processing complete. Files with multiple categories saved to {output_txt_file}')