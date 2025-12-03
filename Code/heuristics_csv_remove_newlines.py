import os
import json
import re
import ijson
import logging
import csv
import gc
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='script_log.log', level=logging.INFO)

# Get a list of all JSON files that start with 'scraped_js'
json_files = [f for f in os.listdir() if f.endswith('.json') and f.startswith('scraped_js')]

# Directory for temporary CSV files
temp_dir = "temp_csv_files"
os.makedirs(temp_dir, exist_ok=True)

def clean_script(script: str) -> str:
    """Removes control characters that could break the CSV format."""
    return script.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()

def categorize_js_script(js_script: str):
    categories = []
    if (re.search(r'fillText|strokeText', js_script) and 
        re.search(r'fillStyle|strokeStyle', js_script) and 
        re.search(r'toDataURL', js_script) and 
        not re.search(r'save|restore|addEventListener', js_script)):
        categories.append("Canvas Fingerprinting")

    if (re.search(r'createDataChannel|createOffer', js_script) and 
        re.search(r'onicecandidate|localDescription', js_script)):
        categories.append("WebRTC Fingerprinting")
    
    if len(re.findall(r'font\s*=\s*', js_script)) > 20 and len(re.findall(r'measureText', js_script)) > 20:
        categories.append("Canvas Font Fingerprinting")
    
    if (re.search(r'createOscillator|createDynamicsCompressor', js_script) and 
        re.search(r'destination|startRendering|oncomplete', js_script)):
        categories.append("AudioContext Fingerprinting")
    
    return categories

def getScriptContent(data):
    script_contents = []
    for item in data:
        content = item.get("content", {})
        for key, scripts in content.items():
            for script in scripts:
                script_content = script.get("script_content", "").strip()
                if script_content:
                    cleaned_script = clean_script(script_content)
                    script_contents.append(cleaned_script)
    return script_contents

def process_json_file(f):
    """Processes a single JSON file and writes results to a temporary CSV file."""
    temp_file = os.path.join(temp_dir, f"{os.path.basename(f)}.csv")
    try:
        with open(temp_file, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow(["script", "categories"])
            
            with open(f, 'r', encoding="utf-8") as fp:
                parser = ijson.items(fp, 'item')  

                for item in parser:
                    scripts = getScriptContent([item])

                    for script in scripts:
                        categories = categorize_js_script(script)
                        if categories:
                            writer.writerow([script, ";".join(categories)])
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from file {f}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing file {f}: {e}")

def merge_csv_files(output_file):
    """Merges all temporary CSV files into the final output CSV file."""
    with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["script", "categories"])
        
        for temp_file in tqdm(os.listdir(temp_dir), desc="Merging CSV files"):
            temp_path = os.path.join(temp_dir, temp_file)
            with open(temp_path, "r", newline="", encoding="utf-8") as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(row)

if __name__ == "__main__":
    for f in tqdm(json_files, desc="Processing JSON files"):
        process_json_file(f)
    
    merge_csv_files("fingerprinting_scripts.csv")
    
    # Clean up temporary files
    for temp_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, temp_file))
    os.rmdir(temp_dir)
    
    gc.collect()
