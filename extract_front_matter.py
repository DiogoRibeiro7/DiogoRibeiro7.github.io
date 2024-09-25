import yaml

def extract_and_print_front_matter(folder: str, file_name: str):
    """
    Extracts and prints the YAML front matter from a Markdown file.

    Args:
        file_name (str): Path to the markdown file.
    """
    try:
        with open(folder + file_name, 'r', encoding='utf-8') as file:
            content = file.read()

            # Check if file starts with YAML front matter
            if content.startswith('---'):
                # Find the end of the front matter (second occurrence of '---')
                front_matter_end = content.find('---', 3)
                if front_matter_end != -1:
                    front_matter = content[3:front_matter_end].strip()
                    parsed_front_matter = yaml.safe_load(front_matter)  # Parse YAML front matter
                    print("Extracted Front Matter:")
                    print(parsed_front_matter)
                else:
                    print("No valid front matter found in the file.")
            else:
                print("File does not contain YAML front matter.")
    
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
folder = './_posts/'
file_name = '2023-01-01-error_coefficientes.md'  # Replace with your file name
extract_and_print_front_matter(folder, file_name)
