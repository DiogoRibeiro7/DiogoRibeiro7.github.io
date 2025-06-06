import os
import re
import yaml
from typing import List

def read_markdown_files_from_folder(folder_path: str) -> List[str]:
    # List all markdown files in the given folder
    return [f for f in os.listdir(folder_path) if f.endswith('.md')]

def extract_frontmatter(file_content: str) -> tuple:
    # Extract the YAML frontmatter from the markdown file using regex
    try:
        # Ensure the content is treated as a raw string to handle escape characters
        frontmatter_match = re.search(r'---\n(.*?)\n---', file_content, re.DOTALL)
        if frontmatter_match:
            frontmatter_str = frontmatter_match.group(1)
            try:
                frontmatter = yaml.safe_load(frontmatter_str)
                return frontmatter, frontmatter_str
            except yaml.YAMLError:
                return {}, ''
    except re.error as e:
        print(f"Regex error: {e}")
    return {}, ''

def clean_tags(frontmatter: dict) -> dict:
    # Ensure the 'tags' key has unique elements, capitalizing the first letter
    if 'tags' in frontmatter and isinstance(frontmatter['tags'], list):
        # Normalize by capitalizing the first letter of each tag and removing duplicates (case-insensitive)
        unique_tags = set()  # To ensure uniqueness
        cleaned_tags = []
        for tag in frontmatter["tags"]:
            capitalized_tag = tag.capitalize()  # Capitalize first letter, lower the rest
            if capitalized_tag not in unique_tags:
                unique_tags.add(capitalized_tag)
                cleaned_tags.append(capitalized_tag)
        frontmatter["tags"] = cleaned_tags
    return frontmatter

def clean_keywords(frontmatter: dict) -> dict:
    # Ensure the 'keywords' key has unique elements, capitalizing the first letter
    if 'keywords' in frontmatter and isinstance(frontmatter['keywords'], list):
        # Normalize by capitalizing the first letter of each tag and removing duplicates (case-insensitive)
        unique_tags = set()  # To ensure uniqueness
        cleaned_tags = []
        for tag in frontmatter["keywords"]:
            capitalized_tag = tag.capitalize()  # Capitalize first letter, lower the rest
            if capitalized_tag not in unique_tags:
                unique_tags.add(capitalized_tag)
                cleaned_tags.append(capitalized_tag)
        frontmatter["keywords"] = cleaned_tags
    return frontmatter

def update_file_content(original_content: str, cleaned_frontmatter: dict) -> str:
    # Replace old frontmatter with cleaned frontmatter
    cleaned_frontmatter_str = yaml.dump(cleaned_frontmatter, default_flow_style=False)
    # Using raw string in regex substitution to avoid escape issues
    new_content = re.sub(r'---\n(.*?)\n---', f'---\n{cleaned_frontmatter_str}---', original_content, flags=re.DOTALL)
    return new_content

def process_markdown_files(folder_path: str):
    markdown_files = read_markdown_files_from_folder(folder_path)

    for md_file in markdown_files:
        file_path = os.path.join(folder_path, md_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            frontmatter, original_frontmatter_str = extract_frontmatter(content)
            if frontmatter:
                cleaned_frontmatter = clean_tags(frontmatter)
                cleaned_frontmatter = clean_keywords(frontmatter)
                new_content = update_file_content(content, cleaned_frontmatter)

                # Write the modified content back to the file if changes were made
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(new_content)

                print(f"Processed file: {md_file}")
        except Exception as e:
            print(f"Error processing file {md_file}: {e}")

folder_path = './_posts'  # Change this to your folder path
process_markdown_files(folder_path)
print(f"Processing complete.")
