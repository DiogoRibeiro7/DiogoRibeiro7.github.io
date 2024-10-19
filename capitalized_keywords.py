import os
import yaml
import re

# Define the folder where the markdown files are stored
folder_path = './_posts'  # Change this to your folder path

# List of stop words to exclude from capitalization
stop_words = {'at', 'vs', 'and', 'or', 'the', 'of', 'in', 'on', 'for', 'to', 'a'}

# Function to capitalize keywords based on your rules
def capitalize_keywords(keywords):
    def capitalize_word(word, first_word=False):
        # Only capitalize if it's not a stop word or it's the first word
        if word in stop_words and not first_word:
            return word
        else:
            return word.capitalize()

    def process_phrase(phrase):
        words = phrase.split()
        # Capitalize each word as per rules, first word always capitalized
        return ' '.join(capitalize_word(word, i == 0) for i, word in enumerate(words))

    return [process_phrase(phrase) for phrase in keywords]

# Function to process each markdown file
def process_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Use regex to extract the front matter (between '---' lines)
    front_matter_match = re.match(r'---(.*?)---', content, re.DOTALL)
    if not front_matter_match:
        print(f"No front matter found in {file_path}")
        return

    front_matter = front_matter_match.group(1)
    
    # Parse the front matter using YAML
    try:
        front_matter_dict = yaml.safe_load(front_matter)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML in {file_path}: {exc}")
        return
    
    # If 'keywords' exists in front matter, process it
    if 'keywords' in front_matter_dict:
        original_keywords = front_matter_dict['keywords']
        updated_keywords = capitalize_keywords(original_keywords)
        front_matter_dict['keywords'] = updated_keywords

        # Replace the front matter in the content
        updated_front_matter = yaml.dump(front_matter_dict, default_flow_style=False)
        
        # Escape backslashes in YAML to avoid issues with re.sub
        updated_front_matter = re.escape(updated_front_matter)
        
        # Rebuild the full content with updated front matter, use re.sub for replacement
        updated_content = re.sub(r'---(.*?)---', f'---\n{updated_front_matter}\n---', content, flags=re.DOTALL)
        
        # Unescape YAML content for the final write back
        updated_content = updated_content.replace(r'\n', '\n').replace(r'\\', '\\')

        # Save the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(updated_content)
        
        print(f"Updated keywords in {file_path}")
    else:
        print(f"No 'keywords' found in {file_path}")

# Function to process all markdown files in the folder
def process_all_markdown_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):  # Check if it's a markdown file
            file_path = os.path.join(folder_path, filename)
            process_markdown_file(file_path)

# Run the function for the specified folder
process_all_markdown_files(folder_path)
