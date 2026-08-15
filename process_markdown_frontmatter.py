import os
import re
import yaml  # You might need to install PyYAML (pip install pyyaml)
import argparse

def process_frontmatter(frontmatter: dict, skip_without_tags_keywords: bool = False):
    """
    Modify the first letter of each tag or keyword in the frontmatter if they exist.

    :param frontmatter: The frontmatter dictionary loaded from YAML.
    :return: Modified frontmatter.
    """
    target_keys = [key for key in ['tags', 'keywords'] if key in frontmatter and isinstance(frontmatter[key], list)]
    if skip_without_tags_keywords and not target_keys:
        return frontmatter, False

    for key in ['tags', 'keywords']:  # Adjust based on the actual fields you want to modify
        if key in frontmatter and isinstance(frontmatter[key], list):
            # Capitalize the first letter of each word in the list
            frontmatter[key] = [str(tag).capitalize() for tag in frontmatter[key]]
        else:
            # If 'tags' or 'keywords' don't exist, just skip
            print(f"'{key}' not found in frontmatter, skipping modification for this field.")
    return frontmatter, True

def process_markdown_file(filepath: str, skip_without_tags_keywords: bool = False):
    """
    Process a markdown file by reading the frontmatter, modifying the tags/keywords if they exist, and saving the file.

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
                # Process the frontmatter to update tags and keywords if they exist
                updated_frontmatter, should_write = process_frontmatter(
                    frontmatter,
                    skip_without_tags_keywords=skip_without_tags_keywords,
                )
                if not should_write:
                    print(f"Skipping file without tags/keywords: {filepath}")
                    return False

                # Replace the original frontmatter in the content
                updated_frontmatter_str = yaml.dump(updated_frontmatter, default_flow_style=False).strip()
                content = content.replace(frontmatter_str, updated_frontmatter_str)

                # Write the updated content back to the file
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(content)
                print(f"Processed file: {filepath}")
                return True
        except yaml.YAMLError as e:
            print(f"Error processing YAML in file {filepath}: {e}")
    else:
        print(f"No frontmatter found in file: {filepath}")
    return False

def process_folder(folder_path: str, skip_without_tags_keywords: bool = False):
    """
    Process all markdown files in the given folder.

    :param folder_path: Path to the folder containing markdown files.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                filepath = os.path.join(root, file)
                process_markdown_file(filepath, skip_without_tags_keywords=skip_without_tags_keywords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process markdown front matter")
    parser.add_argument("--path", default="./_posts", help="Target folder")
    parser.add_argument("--skip-without-tags-keywords", action="store_true", help="Skip files that do not define tags or keywords")
    args = parser.parse_args()
    process_folder(args.path, skip_without_tags_keywords=args.skip_without_tags_keywords)
