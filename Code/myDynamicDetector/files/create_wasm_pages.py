import os
import json
import re
import argparse

def main(source_dir: str, new_json_path: str, output_dir: str):
    """
    Copies HTML files from a source directory to an output directory,
    replacing the content of their <script> tags based on data from a
    new JSON file.

    The script matches HTML filenames (e.g., "10.html") to keys in the
    JSON file (e.g., "10").

    Args:
        source_dir: The directory containing the original HTML files.
        new_json_path: The path to the JSON file with the updated script content.
        output_dir: The directory where the modified HTML files will be saved.
    """
    # 1. Load the new JSON data into a dictionary for fast lookups
    try:
        with open(new_json_path, 'r', encoding='utf-8') as f:
            new_data = json.load(f)
            new_data = {k: v for d in new_data for k, v in d.items()}
        print(f"Successfully loaded new script data from '{new_json_path}'.")
    except FileNotFoundError:
        print(f"Error: The JSON file '{new_json_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{new_json_path}' is not a valid JSON file.")
        return

    # 2. Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved in the '{output_dir}' directory.")

    # 3. Get the list of HTML files to process from the source directory
    try:
        html_files = [f for f in os.listdir(source_dir) if f.endswith('.html')]
        if not html_files:
            print(f"Warning: No .html files were found in '{source_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: The source directory '{source_dir}' does not exist.")
        return

    # 4. Process each HTML file
    print("\nProcessing files...")
    files_processed = 0
    files_skipped = 0
    for filename in html_files:
        # Extract the key from the filename (e.g., '1.html' -> '1')
        key = os.path.splitext(filename)[0]
        source_filepath = os.path.join(source_dir, filename)
        output_filepath = os.path.join(output_dir, filename)

        # Check if this key exists in our new JSON data
        if key in new_data and 'content' in new_data[key]:
            # Get the new script content from the JSON
            new_script_content = new_data[key]['content']

            try:
                # Read the original HTML file content
                with open(source_filepath, 'r', encoding='utf-8') as f_in:
                    original_html = f_in.read()

                # Regex to find content between <script> tags.
                # re.DOTALL makes '.' match newlines, which is crucial for scripts.
                script_pattern = re.compile(r'(<script.*?>)(.*?)(</script>)', re.DOTALL)
                
                # Check if a script tag actually exists in the file
                if script_pattern.search(original_html):
                    # Replace the content inside the first script tag found
                    modified_html = script_pattern.sub(
                        # Use a lambda to rebuild the tag with new content inside
                        lambda m: m.group(1) + f"\n{new_script_content}\n" + m.group(3),
                        original_html,
                        count=1  # Only replace the first occurrence
                    )

                    # Write the modified content to the new file
                    with open(output_filepath, 'w', encoding='utf-8') as f_out:
                        f_out.write(modified_html)
                    
                    print(f"  -> Updated '{filename}'")
                    files_processed += 1
                else:
                    # If no script tag is found, just copy the file as-is
                    with open(source_filepath, 'r', encoding='utf-8') as f_in, \
                         open(output_filepath, 'w', encoding='utf-8') as f_out:
                        f_out.write(f_in.read())
                    print(f"  -> No <script> tag in '{filename}'. Copied without changes.")
                    files_skipped += 1

            except Exception as e:
                print(f"  -> ❌ Error processing '{filename}': {e}")
                files_skipped += 1
        else:
            print(f"  -> ⚠️  Warning: Key '{key}' (from '{filename}') not found in JSON. Copied without changes.")
            # Copy the file without modification if no key is found
            with open(source_filepath, 'r', encoding='utf-8') as f_in, \
                 open(output_filepath, 'w', encoding='utf-8') as f_out:
                f_out.write(f_in.read())
            files_skipped += 1

    print(f"\n🎉 Done! {files_processed} files updated, {files_skipped} files copied/skipped.")


if __name__ == '__main__':
    # --- CONFIGURE YOUR SETTINGS HERE ---
    argparse_parser = argparse.ArgumentParser(
        description="Update HTML files' <script> content based on a new JSON file."
    )
    argparse_parser.add_argument(
        '--source_dir', type=str, required=True,
        help='The folder containing the original HTML pages you want to update.'
    )
    argparse_parser.add_argument(
        '--new_json', type=str, required=True,
        help='The new JSON file with the updated "content" strings.'
    )
    argparse_parser.add_argument(
        '--output_dir', type=str, required=True,
        help='The folder where the new version of the pages will be saved.'
    )
    args = argparse_parser.parse_args()
    main(args.source_dir, args.new_json, args.output_dir)
