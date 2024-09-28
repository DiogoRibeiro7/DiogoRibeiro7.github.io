import os
import re

# Function to extract code snippets and their languages
def extract_code_snippets(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

        # Regular expression to find code blocks with optional language declaration
        # Capture the language if present and the code content
        code_snippets = re.findall(r'```(\w+)?\n(.*?)\n```', content, re.DOTALL)

        # Dictionary to store snippets by language
        snippets_by_language = {}

        for snippet in code_snippets:
            language = snippet[0] if snippet[0] else 'unknown'  # If no language is specified, categorize as 'unknown'
            code = snippet[1].strip()

            if language not in snippets_by_language:
                snippets_by_language[language] = {"tasks": [], "keywords": []}
            
            # Add the snippet to both 'tasks' and 'keywords' keys for the language
            snippets_by_language[language]["tasks"].append(code)
            snippets_by_language[language]["keywords"].append(code)
        
        return snippets_by_language

# Function to iterate over all markdown files in a folder and extract code snippets
def search_markdown_files(folder_path: str):
    all_snippets_by_language = {}

    # Walk through all files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                snippets_by_language = extract_code_snippets(file_path)

                # Merge snippets from the current file into the global result dictionary
                for language, snippets in snippets_by_language.items():
                    if language not in all_snippets_by_language:
                        all_snippets_by_language[language] = {"tasks": [], "keywords": []}

                    all_snippets_by_language[language]["tasks"].extend(snippets["tasks"])
                    all_snippets_by_language[language]["keywords"].extend(snippets["keywords"])
    
    return all_snippets_by_language

# Example usage
folder_path = './_posts'  # Update this path to your markdown folder
all_snippets_by_language = search_markdown_files(folder_path)

# Print the organized snippets by language
for language, snippets in all_snippets_by_language.items():
    print(f"Language: {language}")
    print(f"Tasks ({len(snippets['tasks'])}):")
    for task in snippets['tasks']:
        print(task)
        print('-' * 40)
    print(f"Keywords ({len(snippets['keywords'])}):")
    for keyword in snippets['keywords']:
        print(keyword)
        print('-' * 40)
    print('\n')
