# Contributing

This project uses several helper scripts to keep the Markdown posts and front matter consistent. These scripts can modify many files across the repository.

Because these updates are often large-scale, test them on a separate branch before merging.

## Running automation scripts

1. Start from an up-to-date branch and create a new working branch:

   ```bash
   git checkout master
   git pull
   git checkout -b automation-update
   ```

2. Run the automation scripts:

   ```bash
   ./run_scripts.sh
   ```

   The scripts may rename files and update front matter, leading to many modified files.

3. Review the changes:

   ```bash
   git status
   git diff
   ```

   Look carefully over the diff before committing because the updates can be large.

4. Commit the desired changes and open a pull request.

Running the scripts on a clean branch helps keep your history tidy and makes code review easier.
