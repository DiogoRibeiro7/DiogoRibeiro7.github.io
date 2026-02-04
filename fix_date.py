import os
import re
import frontmatter
import argparse

# TODO: Add a dry-run option to report changes without writing files.
def extract_date_from_filename(filename):
    # Assuming the filename format is 'YYYY-MM-DD-some-title.md'
    match = re.match(r'(\d{4}-\d{2}-\d{2})-', filename)
    if match:
        return match.group(1)
    return None

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

    # Write the updated content back to the markdown file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(frontmatter.dumps(post))

def process_markdown_files_in_directory(directory):
    # Process all markdown files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            filepath = os.path.join(directory, filename)
            process_markdown_file(filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix dates in markdown front matter")
    parser.add_argument("--path", default="./_posts", help="Target folder")
    args = parser.parse_args()
    process_markdown_files_in_directory(args.path)
