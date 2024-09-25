import os
import yaml  # to parse YAML front matter

def extract_front_matter(md_file_path: str) -> dict:
    """
    Extracts YAML front matter from a Markdown file.
    
    Args:
        md_file_path (str): Path to the markdown file.

    Returns:
        dict: Parsed front matter as a dictionary, or None if not found.
    """
    with open(md_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # Check if file starts with YAML front matter
        if content.startswith('---'):
            # Front matter ends with another '---'
            front_matter_end = content.find('---', 3)
            if front_matter_end != -1:
                front_matter = content[3:front_matter_end].strip()
                return yaml.safe_load(front_matter)  # Parse YAML front matter
                
    return None

def check_front_matter(folder_path: str, output_file: str):
    """
    Checks if 'summary' and 'keywords' keys are present in the front matter of Markdown files
    and saves the output to a text file only if any of the keys are missing.

    Args:
        folder_path (str): Path to the folder containing markdown files.
        output_file (str): Path to the output text file where results will be saved.
    """
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.md'):  # Check only markdown files
                file_path = os.path.join(folder_path, file_name)
                front_matter = extract_front_matter(file_path)
                
                if front_matter is None:
                    out_file.write(f"File '{file_name}' does not contain a valid front matter.\n\n")
                else:
                    has_summary = 'summary' in front_matter
                    has_keywords = 'keywords' in front_matter

                    # Write only if any key is missing
                    if not has_summary or not has_keywords:
                        out_file.write(f"File '{file_name}':\n")
                        out_file.write(f"  - Summary present: {has_summary}\n")
                        out_file.write(f"  - Keywords present: {has_keywords}\n")
                        out_file.write("\n")

# Example usage
folder_path = './_posts'  # Replace with the actual folder path
output_file = "front_matter_report.txt"  # Replace with the desired output file path
check_front_matter(folder_path, output_file)
