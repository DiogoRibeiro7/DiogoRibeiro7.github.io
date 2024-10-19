#!/bin/bash

python markdown_file_processor.py  
python fix_frontmatter.py 
python search_code_snippets.py    
# python process_markdown_frontmatter.py  
python rename_files_spaces.py
python markdown_frontmatter_cleanup.py
