import os
import json
import re
import ijson
import logging
import gc
from tqdm import tqdm

# Set up logging
logging.basicConfig(filename='script_log.log', level=logging.INFO)

# Get a list of all JSON files that start with 'scraped_js'
json_files = [f for f in os.listdir() if f.endswith('.json') and f.startswith('scraped_js')]

# Directory for temporary JSON files
temp_dir = "temp_json_files"
os.makedirs(temp_dir, exist_ok=True)

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
                    script_contents.append(script_content)
    return script_contents

def process_json_file(f):
    """Processes a single JSON file and writes results to a temporary JSON file."""
    temp_file = os.path.join(temp_dir, f"{os.path.basename(f)}.json")
    temp_data = []
    try:
        with open(f, 'r', encoding="utf-8") as fp:
            parser = ijson.items(fp, 'item')  
            
            for item in parser:
                scripts = getScriptContent([item])

                for script in scripts:
                    categories = categorize_js_script(script)
                    if categories:
                        temp_data.append({"script": script, "categories": categories})
        
        with open(temp_file, "w", encoding="utf-8") as jsonfile:
            json.dump(temp_data, jsonfile, indent=4)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from file {f}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing file {f}: {e}")

def merge_json_files(output_file):
    """Merges all temporary JSON files into the final output JSON file."""
    merged_data = []
    
    for temp_file in tqdm(os.listdir(temp_dir), desc="Merging JSON files"):
        temp_path = os.path.join(temp_dir, temp_file)
        try:
            with open(temp_path, "r", encoding="utf-8") as infile:
                data = json.load(infile)
                merged_data.extend(data)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from temp file {temp_file}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error reading temp file {temp_file}: {e}")
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(merged_data, outfile, indent=4)

if __name__ == "__main__":
    for f in tqdm(json_files, desc="Processing JSON files"):
        process_json_file(f)
    
    merge_json_files("fingerprinting_scripts.json")
    
    # Clean up temporary files
    for temp_file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, temp_file))
    os.rmdir(temp_dir)
    
    gc.collect()
