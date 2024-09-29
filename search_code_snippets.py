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
    # Updated regular expression to capture more language identifiers
    code_snippets = re.findall(r'```([a-zA-Z0-9+_-]*)\n(.*?)\n```', content, re.DOTALL)

    # Dictionary to store snippets by language
    snippets_by_language = {}

    for snippet in code_snippets:
        language = snippet[0] if snippet[0] else 'unknown'  # If no language is specified, categorize as 'unknown'
        code = snippet[1].strip()

        if language not in snippets_by_language:
            snippets_by_language[language] = []

        snippets_by_language[language].append(code)

    return snippets_by_language

# Function to update front matter with tasks and keywords, appending new languages
def update_front_matter(front_matter: dict, snippets_by_language: dict):
    detected_languages = list(snippets_by_language.keys())

    # Function to safely convert a string of tags or keywords into a list
    def ensure_list(field):
        if isinstance(field, str):
            # Convert comma-separated string to list
            return [item.strip() for item in field.split(',')]
        return field

    # Ensure 'tags' is a list and append new languages if not already present
    if 'tags' in front_matter:
        front_matter['tags'] = ensure_list(front_matter['tags'])
        for lang in detected_languages:
            if lang not in front_matter['tags']:
                front_matter['tags'].append(lang)
    else:
        front_matter['tags'] = detected_languages

    # Only update 'keywords' if it already exists in the front matter
    if 'keywords' in front_matter:
        front_matter['keywords'] = ensure_list(front_matter['keywords'])
        for lang in detected_languages:
            if lang not in front_matter['keywords']:
                front_matter['keywords'].append(lang)

    # Do not add 'keywords' if they are not present in the original front matter
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

                # Update the front matter by appending new languages to tags and existing keywords
                updated_front_matter = update_front_matter(front_matter, snippets_by_language)

                # Create the new content with updated front matter
                new_front_matter = yaml.dump(updated_front_matter, default_flow_style=False).strip()
                new_content = f"---\n{new_front_matter}\n---\n\n{content_without_front_matter}"

                # Write the updated content back to the file
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                print(f"Updated front matter in {file}")


# Example usage
folder_path = './_posts'  # Update this path to your markdown folder
process_markdown_files(folder_path)
