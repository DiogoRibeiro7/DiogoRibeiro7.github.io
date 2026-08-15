import os
import re
import argparse

def contains_code_block(content: str) -> bool:
    return bool(re.search(r"```.*?```", content, re.DOTALL))

def replace_latex_syntax_in_file(file_path: str, skip_code_blocks: bool = False):
    """
    This function reads a markdown file, finds LaTeX delimiters and replaces them 
    with double dollar signs for compatibility with a different LaTeX rendering system.

    Args:
        file_path (str): The path to the markdown file that needs to be processed.

    Returns:
        None
    """
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    if skip_code_blocks and contains_code_block(content):
        print(f"Skipping file with code block: {file_path}")
        return False

    # Define the patterns to be replaced
    updated_content = re.sub(r'\\\[', '$$', content)  # Replaces \[ with $$
    updated_content = re.sub(r'\\\]', '$$', updated_content)  # Replaces \] with $$
    updated_content = re.sub(r'\\\(', '$$', updated_content)  # Replaces \( with $$
    updated_content = re.sub(r'\\\)', '$$', updated_content)  # Replaces \) with $$

    if updated_content == content:
        return False

    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)
    return True

def process_markdown_files_in_folder(folder_path: str, skip_code_blocks: bool = False):
    """
    Processes all markdown files in a given folder, replacing LaTeX delimiters
    according to the replacement rules.

    Args:
        folder_path (str): Path to the folder containing markdown files.

    Returns:
        None
    """
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Process only markdown files
        if filename.endswith('.md'):
            file_path = os.path.join(folder_path, filename)
            print(f'Processing file: {file_path}')
            replace_latex_syntax_in_file(file_path, skip_code_blocks=skip_code_blocks)
            print(f'Finished processing file: {file_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace LaTeX delimiters in markdown files")
    parser.add_argument("--path", default="./_posts", help="Target folder")
    parser.add_argument("--skip-code-blocks", action="store_true", help="Skip files containing fenced code blocks")
    args = parser.parse_args()
    process_markdown_files_in_folder(args.path, skip_code_blocks=args.skip_code_blocks)
