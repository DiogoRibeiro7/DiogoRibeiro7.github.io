import os
import re
import frontmatter
import random

TOTAL_FILES = 20

def extract_date_from_filename(filename):
    # Assuming the filename format is 'YYYY-MM-DD-some-title.md'
    match = re.match(r'(\d{4}-\d{2}-\d{2})-', filename)
    if match:
        return match.group(1)
    return None

def ensure_single_newline(content):
    # Check if content already ends with a newline
    if not content.endswith('\n'):
        return content + '\n'
    return content

def process_markdown_file(filepath):
    # Load the markdown file and parse its frontmatter
    with open(filepath, 'r', encoding='utf-8') as f:
        post = frontmatter.load(f)
    
    # Extract the date from the filename
    filename = os.path.basename(filepath)
    file_date = extract_date_from_filename(filename)

    if not file_date:
        print(f"Could not extract date from filename: {filename}")
        return
    
    # Check if the frontmatter already contains a date
    frontmatter_date = post.get('date')

    if frontmatter_date:
        # If the date is different, update it
        if frontmatter_date != file_date:
            print(f"Updating date in {filename}: {frontmatter_date} -> {file_date}")
            post['date'] = file_date
    else:
        # If no date exists, add it
        print(f"Adding date to {filename}: {file_date}")
        post['date'] = file_date

    # Check if 'toc' is present and is True, and change it to False
    if post.get('toc') is True:
        print(f"Changing 'toc' to False in {filename}")
        post['toc'] = False
        
    # Insert 'header' block if not already present
    random_number = random.randint(1, TOTAL_FILES)
    if 'header' not in post:
        print(f"Inserting 'header' block in {filename}")
        post['header'] = {
            'image': f'/assets/images/data_science_{random_number}.jpg',
            'overlay_image': f'/assets/images/data_science_{random_number}.jpg',
            'teaser': f'/assets/images/data_science_{random_number}.jpg',
            'show_overlay_excerpt': False,  # Add excerpt key with value False when header is inserted
            'twitter_image': f'/assets/images/data_science_{random_number}.jpg',
            'og_image': f'/assets/images/data_science_{random_number}.jpg'
        }
    else:
        # If 'header' exists, check if 'excerpt' key is present
        if 'show_overlay_excerpt' not in post['header']:
            print(f"Adding 'excerpt' key to 'header' in {filename}")
            post['header']['show_overlay_excerpt'] = False
        else:
            print(f"'excerpt' key already present in 'header' in {filename}, skipping.")
        if 'twitter_image' not in post['header']:
            print(f"Adding 'twitter_image' key to 'header' in {filename}")
            post['header']['twitter_image'] = f'/assets/images/data_science_{random_number}.jpg'
        if 'og_image' not in post['header']:
            print(f"Adding 'og_image' key to 'header' in {filename}")
            post['header']['og_image'] = f'/assets/images/data_science_{random_number}.jpg'
        
    if not "seo_type" in post:
        post["seo_type"] = "article"

    # Dump the updated content and ensure it ends with a single newline
    content = frontmatter.dumps(post)
    content = ensure_single_newline(content)

    # Write the updated content back to the markdown file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def process_markdown_files_in_directory(directory):
    # Process all markdown files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            process_markdown_file(filepath)

# Example usage:
directory_path = './_posts'  # Change to your directory path
process_markdown_files_in_directory(directory_path)
