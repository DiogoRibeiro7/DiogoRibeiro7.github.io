import os
import re
import argparse

def replace_latex_syntax_in_file(file_path: str):
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

    # Define the patterns to be replaced
    content = re.sub(r'\\\[', '$$', content)  # Replaces \[ with $$
    content = re.sub(r'\\\]', '$$', content)  # Replaces \] with $$
    content = re.sub(r'\\\(', '$$', content)  # Replaces \( with $$
    content = re.sub(r'\\\)', '$$', content)  # Replaces \) with $$

    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process_markdown_files_in_folder(folder_path: str):
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
            replace_latex_syntax_in_file(file_path)
            print(f'Finished processing file: {file_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replace LaTeX delimiters in markdown files")
    parser.add_argument("--path", default="./_posts", help="Target folder")
    args = parser.parse_args()
    process_markdown_files_in_folder(args.path)
