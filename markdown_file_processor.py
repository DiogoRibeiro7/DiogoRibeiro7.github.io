import os
import re
import string

# List of stop words to remove from file names
STOP_WORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "that",
    "this",
    "is",
    "in",
    "on",
    "with",
    "of",
    "for",
    "to",
    "by",
    "at",
    "as",
    "from",
    "it",
    "be",
    "are",
    "was",
    "were",
    "will",
    "has",
    "have",
}

def rename_markdown_file(file_path: str) -> str:
    """
    Renames the markdown file so that the name part after the date is in lowercase,
    spaces are replaced with underscores, stop words are removed, and special symbols
    are excluded from the title.

    Args:
        file_path (str): The original file path of the markdown file.

    Returns:
        str: The new file path after renaming.
    """
    # Extract filename from the path
    filename = os.path.basename(file_path)

    # Split the filename to get the date and the name parts
    date_part, name_part = filename.split("-", 3)[:3], filename.split("-", 3)[3]
    name_part = os.path.splitext(name_part)[0]  # Remove the .md extension

    # Split name into words, remove stop words, and replace spaces with underscores
    name_words = name_part.lower().split("_")
    filtered_name = [word for word in name_words if word not in STOP_WORDS]

    # Remove special symbols from each word
    filtered_name = [word.translate(str.maketrans('', '', string.punctuation)) for word in filtered_name]

    # Construct the formatted name by joining with underscores
    formatted_name = "_".join(filtered_name)

    # Construct the new filename
    new_filename = f"{'-'.join(date_part)}-{formatted_name}.md"
    new_file_path = os.path.join(os.path.dirname(file_path), new_filename)

    # Rename the file
    os.rename(file_path, new_file_path)
    print(f"Renamed '{filename}' to '{new_filename}'")
    return new_file_path

def replace_latex_syntax_in_file(file_path: str):
    """
    Reads a markdown file, finds LaTeX delimiters, and replaces them
    with double dollar signs for compatibility with a different LaTeX rendering system.

    Args:
        file_path (str): The path to the markdown file that needs to be processed.

    Returns:
        None
    """
    # Read the content of the file
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Define the patterns to be replaced
    content = re.sub(r"\\\\\[", "$$", content)  # Replaces \[ with $$
    content = re.sub(r"\\\\\]", "$$", content)  # Replaces \] with $$
    content = re.sub(r"\\\\\(", "$$", content)  # Replaces \( with $$
    content = re.sub(r"\\\\\)", "$$", content)  # Replaces \) with $$

    # Write the updated content back to the file
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)

def process_markdown_files_in_folder(folder_path: str):
    """
    Processes all markdown files in a given folder, renaming them and
    replacing LaTeX delimiters according to the replacement rules.

    Args:
        folder_path (str): Path to the folder containing markdown files.

    Returns:
        None
    """
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Process only markdown files
        if filename.endswith(".md"):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing file: {file_path}")

            # Rename the file
            new_file_path = rename_markdown_file(file_path)

            # Replace LaTeX syntax in the renamed file
            replace_latex_syntax_in_file(new_file_path)

            print(f"Finished processing file: {new_file_path}")


# Path to the folder containing markdown files
folder_path = "./_posts"
process_markdown_files_in_folder(folder_path)
