#!/bin/bash

TARGET="./_posts"

python markdown_file_processor.py --path "$TARGET"
python fix_frontmatter.py --path "$TARGET"
python search_code_snippets.py --path "$TARGET"
# python process_markdown_frontmatter.py --path "$TARGET"
python rename_files_spaces.py --path "$TARGET"
python markdown_frontmatter_cleanup.py --path "$TARGET"
