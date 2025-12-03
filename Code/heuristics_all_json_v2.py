import os
import json
import re
from tqdm import tqdm
import logging
import time
import psutil
import signal
import sys
import gc

# Set up logging for debugging purposes
logging.basicConfig(filename='script_log.log', level=logging.INFO)

# Function to check memory and CPU usage and handle failsafes
def check_system_limits(memory_limit_mb=500, cpu_limit_percent=80):
    process = psutil.Process(os.getpid())

    # Check memory usage
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
    if memory_usage > memory_limit_mb:
        logging.warning(f'Memory usage exceeded limit: {memory_usage} MB. Exiting script.')
        sys.exit("Memory limit exceeded.")

    # Check CPU usage
    cpu_usage = process.cpu_percent(interval=1)
    if cpu_usage > cpu_limit_percent:
        logging.warning(f'CPU usage exceeded limit: {cpu_usage}%. Sleeping to reduce CPU load.')
        time.sleep(5)  # Sleep to reduce CPU load

# Function to handle termination signals (e.g., SIGTERM, SIGINT)
def handle_signal(signal_number, frame):
    logging.info("Received termination signal. Cleaning up and exiting.")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

# Get a list of all files in the current directory
files = os.listdir()

def categorize_js_script(js_script: str):
    categories = []

    # Canvas Fingerprinting - Checks for fillText, strokeText, fillStyle, strokeStyle, toDataURL, save, restore, and addEventListener
    if (re.search(r'fillText|strokeText', js_script) and 
        re.search(r'fillStyle|strokeStyle', js_script) and 
        re.search(r'toDataURL', js_script) and 
        not re.search(r'save|restore|addEventListener', js_script)):
        categories.append("Canvas Fingerprinting")

    # WebRTC Fingerprinting - Checks for createDataChannel, createOffer, onicecandidate, localDescription
    if (re.search(r'createDataChannel|createOffer', js_script) and 
        re.search(r'onicecandidate|localDescription', js_script)):
        categories.append("WebRTC Fingerprinting")
    
    # Canvas Font Fingerprinting - Checks for font property set multiple times and measureText called more than 20 times
    if len(re.findall(r'font\s*=', js_script)) > 20 and len(re.findall(r'measureText', js_script)) > 20:
        categories.append("Canvas Font Fingerprinting")
    
    # AudioContext Fingerprinting - Checks for createOscillator, createDynamicsCompressor, destination, startRendering, oncomplete
    if (re.search(r'createOscillator|createDynamicsCompressor', js_script) and 
        re.search(r'destination|startRendering|oncomplete', js_script)):
        categories.append("AudioContext Fingerprinting")
    
    # Return the categories that match
    return categories

def getScriptContent(data):
    # Extract script_content values
    script_contents = []
    
    for item in data:
        content = item.get("content", {})
        for key, scripts in content.items():
            for script in scripts:
                # Check if script_content exists and is not empty
                if script.get("script_content"):
                    script_contents.append(script["script_content"])
                    
    return script_contents

# Filter for JSON files that start with 'scraped_js'
json_files = [f for f in files if f.endswith('.json') and f.startswith('scraped_js')]

# Process each JSON file and append results to the output file
output_file = "fingerprinting_scripts.json"

for f in tqdm(json_files):
    try:
        # Open the file one by one and load it incrementally
        with open(f) as fp:
            scripts = json.load(fp)
            scripts = getScriptContent(scripts)

            # Process each script and categorize it
        for script in scripts:
            check_system_limits()  # Check for memory and CPU limits before processing each script
            categories = categorize_js_script(script)
            if categories:
                result = {"script": script, "categories": categories}
                # Open the output file in append mode to add data incrementally
                with open(output_file, "a") as ff:
                    json.dump([result], ff, indent=4)  # Append the data as a new entry
                    ff.write(",\n")  # Add a separator for the next entry
            
            # Manually trigger garbage collection to free memory after processing each script
            gc.collect()

    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from file {f}: {e}")
        continue  # Skip the file if it's malformed
    except Exception as e:
        logging.error(f"Unexpected error processing file {f}: {e}")
        continue

# Final cleanup to ensure the file ends with a proper JSON array
with open(output_file, "rb+") as ff:
    ff.seek(-2, os.SEEK_END)  # Go to the second to last character
    ff.truncate()  # Remove the last comma
    ff.write(b"\n]")  # Close the JSON array
