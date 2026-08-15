import os
import yaml  # to parse YAML front matter
import argparse
import json

def extract_front_matter(md_file_path: str) -> dict:
    """
    Extracts YAML front matter from a Markdown file.
    
    Args:
        md_file_path (str): Path to the markdown file.

    Returns:
        dict: Parsed front matter as a dictionary, or None if not found.
    """
    with open(md_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # Check if file starts with YAML front matter
        if content.startswith('---'):
            # Front matter ends with another '---'
            front_matter_end = content.find('---', 3)
            if front_matter_end != -1:
                front_matter = content[3:front_matter_end].strip()
                return yaml.safe_load(front_matter)  # Parse YAML front matter
                
    return None

def check_front_matter(folder_path: str, output_file: str, output_format: str = "text"):
    """
    Checks if 'summary' and 'keywords' keys are present in the front matter of Markdown files
    and saves the output to a text file only if any of the keys are missing.

    Args:
        folder_path (str): Path to the folder containing markdown files.
        output_file (str): Path to the output file where results will be saved.
        output_format (str): Output format: "text" or "json".
    """
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.md'):  # Check only markdown files
            file_path = os.path.join(folder_path, file_name)
            front_matter = extract_front_matter(file_path)

            if front_matter is None:
                results.append({
                    "file": file_name,
                    "valid_front_matter": False,
                    "summary_present": False,
                    "keywords_present": False,
                })
            else:
                has_summary = 'summary' in front_matter
                has_keywords = 'keywords' in front_matter

                if not has_summary or not has_keywords:
                    results.append({
                        "file": file_name,
                        "valid_front_matter": True,
                        "summary_present": has_summary,
                        "keywords_present": has_keywords,
                    })

    with open(output_file, 'w', encoding='utf-8') as out_file:
        if output_format == "json":
            json.dump(results, out_file, indent=2)
            out_file.write("\n")
        else:
            for result in results:
                if not result["valid_front_matter"]:
                    out_file.write(f"File '{result['file']}' does not contain a valid front matter.\n\n")
                else:
                    out_file.write(f"File '{result['file']}':\n")
                    out_file.write(f"  - Summary present: {result['summary_present']}\n")
                    out_file.write(f"  - Keywords present: {result['keywords_present']}\n")
                    out_file.write("\n")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check markdown files for summary and keywords")
    parser.add_argument("--path", default="./_posts", help="Target folder")
    parser.add_argument("--output", default="front_matter_report.txt", help="Output file")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()
    check_front_matter(args.path, args.output, args.format)
