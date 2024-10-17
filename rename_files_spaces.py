import os


def rename_files_in_folder(directory: str) -> None:
    """
    Rename files in the given directory by replacing spaces with underscores.

    Parameters:
    directory (str): The path to the directory containing files to be renamed.
    """
    try:
        # Verify if the provided directory exists
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"The path '{directory}' is not a valid directory.")

        # Loop through each file in the directory
        for filename in os.listdir(directory):
            old_path = os.path.join(directory, filename)
            # Ensure we are working with files only, skip directories
            if os.path.isfile(old_path):
                # Replace spaces in the filename with underscores
                new_filename = filename.replace(" ", "_")
                new_path = os.path.join(directory, new_filename)
                # Rename only if the new filename differs from the old one
                if old_path != new_path:
                    os.rename(old_path, new_path)
                    print(f"Renamed '{filename}' to '{new_filename}'")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # You can change the path below to point to your folder
    folder_path: str = './_posts'
    rename_files_in_folder(folder_path)
