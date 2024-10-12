import os
import re
import yaml  # You might need to install PyYAML (pip install pyyaml)

def process_frontmatter(frontmatter: dict):
    """
    Modify the first letter of each tag or keyword in the frontmatter.

    :param frontmatter: The frontmatter dictionary loaded from YAML.
    :return: Modified frontmatter.
    """
    for key in ['tags', 'keywords']:  # Adjust based on the actual fields you want to modify
        if key in frontmatter and isinstance(frontmatter[key], list):
            frontmatter[key] = [tag.capitalize() for tag in frontmatter[key]]
    return frontmatter

def process_markdown_file(filepath: str):
    """
    Process a markdown file by reading the frontmatter, modifying the tags/keywords, and saving the file.

    :param filepath: Path to the markdown file.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    # Find frontmatter (YAML between ---)
    frontmatter_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
    if frontmatter_match:
        frontmatter_str = frontmatter_match.group(1)
        
        try:
            frontmatter = yaml.safe_load(frontmatter_str)
            if isinstance(frontmatter, dict):
                # Process the frontmatter to update tags and keywords
                updated_frontmatter = process_frontmatter(frontmatter)

                # Replace the original frontmatter in the content
                updated_frontmatter_str = yaml.dump(updated_frontmatter, default_flow_style=False).strip()
                content = content.replace(frontmatter_str, updated_frontmatter_str)

                # Write the updated content back to the file
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(content)
                print(f"Processed file: {filepath}")
        except yaml.YAMLError as e:
            print(f"Error processing YAML in file {filepath}: {e}")
    else:
        print(f"No frontmatter found in file: {filepath}")

def process_folder(folder_path: str):
    """
    Process all markdown files in the given folder.

    :param folder_path: Path to the folder containing markdown files.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                process_markdown_file(filepath)


# Specify the folder path containing markdown files
folder_path = './_posts'  # Replace with the actual folder path
process_folder(folder_path)
