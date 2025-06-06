#!/bin/bash

TARGET="./_posts"

python3 markdown_file_processor.py --path "$TARGET"
python3 fix_frontmatter.py --path "$TARGET"
python3 search_code_snippets.py --path "$TARGET"
# python3 process_markdown_frontmatter.py --path "$TARGET"
python3 rename_files_spaces.py --path "$TARGET"
python3 markdown_frontmatter_cleanup.py --path "$TARGET"
