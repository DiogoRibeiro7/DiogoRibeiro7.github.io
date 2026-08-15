import yaml
import argparse
import json
import os

def format_front_matter(front_matter: dict, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(front_matter, indent=2)
    if output_format == "yaml":
        return yaml.safe_dump(front_matter, default_flow_style=False, sort_keys=False).strip()
    return str(front_matter)

def extract_and_print_front_matter(folder: str, file_name: str, output_format: str = "plain"):
    """
    Extracts and prints the YAML front matter from a Markdown file.

    Args:
        file_name (str): Path to the markdown file.
    """
    try:
        file_path = file_name if os.path.isabs(file_name) else os.path.join(folder, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # Check if file starts with YAML front matter
            if content.startswith('---'):
                # Find the end of the front matter (second occurrence of '---')
                front_matter_end = content.find('---', 3)
                if front_matter_end != -1:
                    front_matter = content[3:front_matter_end].strip()
                    parsed_front_matter = yaml.safe_load(front_matter)  # Parse YAML front matter
                    print(format_front_matter(parsed_front_matter, output_format))
                else:
                    print("No valid front matter found in the file.")
            else:
                print("File does not contain YAML front matter.")
    
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and print front matter from a markdown file")
    parser.add_argument("--path", default="./_posts/", help="Folder containing the markdown file")
    parser.add_argument("--file", default="2023-01-01-error_coefficientes.md", help="Markdown file name")
    parser.add_argument("--format", choices=["plain", "yaml", "json"], default="plain", help="Output format")
    args = parser.parse_args()
    extract_and_print_front_matter(args.path, args.file, args.format)
