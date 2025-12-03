import json
import os
from html_generator_regex import html_generator_regex
import argparse


def create_html_from_json(json_path: str, output_dir: str):
    """
    Loads a JSON file, processes each item, and saves generated HTML.
    
    Args:
        json_path: The path to the input JSON file.
        output_dir: The name of the folder to save HTML files in.
    """
    # Create the output directory if it doesn't already exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Open and load the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ Successfully loaded '{json_path}'. Processing files...")
        merged_dict = {k: v for d in data for k, v in d.items()}

        # Iterate over each key-value pair in the JSON data
        for key, value_obj in merged_dict.items():
            if 'content' in value_obj:
                # 1. Get the script content from the 'content' key
                script_content = value_obj['content']
                
                # 2. Pass the script to your generator function
                generated_html = html_generator_regex(script_content)
                
                # 3. Define the output file name and path
                output_filename = f"{key}.html"
                output_filepath = os.path.join(output_dir, output_filename)
                
                # 4. Save the generated HTML to the new file
                with open(output_filepath, 'w', encoding='utf-8') as f_out:
                    f_out.write(generated_html)
                
                print(f"  -> Saved '{output_filepath}'")
            else:
                print(f"  -> ⚠️  Warning: Key '{key}' has no 'content' field. Skipped.")

        print("\n All files processed successfully!")

    except FileNotFoundError:
        print(f" Error: The file '{json_path}' was not found.")
    except json.JSONDecodeError:
        print(f" Error: The file '{json_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate HTML files from a JSON file containing script content.")
    parser.add_argument('--json', type=str, help='Path to the input JSON file.')
    parser.add_argument('--output', type=str, help='Output directory for HTML files.')
    args = parser.parse_args()
    create_html_from_json(args.json, args.output)