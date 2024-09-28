import os
import re
import yaml

# Function to extract front matter from markdown file
def extract_front_matter(content: str):
    front_matter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if front_matter_match:
        front_matter = front_matter_match.group(1)
        yaml_data = yaml.safe_load(front_matter)
        content_without_front_matter = content[front_matter_match.end():].strip()
        return yaml_data, content_without_front_matter
    else:
        return {}, content  # Return empty dict if no front matter found

# Function to extract code snippets and their languages
def extract_code_snippets(content: str):
    # Regular expression to find code blocks with optional language declaration
    code_snippets = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)
    
    # Dictionary to store snippets by language
    snippets_by_language = {}

    for snippet in code_snippets:
        language = snippet[0] if snippet[0] else 'unknown'  # If no language is specified, categorize as 'unknown'
        code = snippet[1].strip()

        if language not in snippets_by_language:
            snippets_by_language[language] = []

        snippets_by_language[language].append(code)

    return snippets_by_language

# Function to update front matter with tasks and keywords
def update_front_matter(front_matter: dict, snippets_by_language: dict):
    front_matter['tasks'] = list(snippets_by_language.keys())  # Add detected languages to 'tasks'
    front_matter['keywords'] = list(snippets_by_language.keys())  # Also add detected languages to 'keywords'
    return front_matter

# Function to iterate over all markdown files in a folder and process them
def process_markdown_files(folder_path: str):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)

                # Read the content of the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract front matter and code snippets
                front_matter, content_without_front_matter = extract_front_matter(content)
                snippets_by_language = extract_code_snippets(content_without_front_matter)

                # Update the front matter with tasks and keywords
                updated_front_matter = update_front_matter(front_matter, snippets_by_language)

                # If front matter exists, replace it
                if front_matter:
                    new_front_matter = yaml.dump(updated_front_matter, default_flow_style=False).strip()
                    new_content = f"---\n{new_front_matter}\n---\n\n{content_without_front_matter}"
                else:
                    # If no front matter, add it
                    new_front_matter = yaml.dump(updated_front_matter, default_flow_style=False).strip()
                    new_content = f"---\n{new_front_matter}\n---\n\n{content}"

                # Write the updated content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                print(f"Updated front matter in {file}")

# Example usage
folder_path = './_posts'  # Update this path to your markdown folder
process_markdown_files(folder_path)
