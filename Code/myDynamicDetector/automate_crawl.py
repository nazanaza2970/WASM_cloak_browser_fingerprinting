import argparse
import os
import subprocess
from tqdm import tqdm
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# The name of the script you want to execute for each link.
# Change this to the actual name of your script.
TARGET_SCRIPT = "crawler_behavior_simulation.py" 

def run_script_on_links(base_url, output_folder):
    """
    Fetches a URL, finds all internal links, and executes a target script for each link.

    Args:
        base_url (str): The starting URL to scrape for links.
        output_folder (str): The directory to save the output files.
    """
    print(f"[*] Starting process for base URL: {base_url}")

    # Ensure the output directory exists
    try:
        os.makedirs(output_folder, exist_ok=True)
        print(f"[*] Output will be saved in: {output_folder}")
    except OSError as e:
        print(f"[!] Error creating directory {output_folder}: {e}")
        return

    # Fetch the HTML content of the base URL
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.RequestException as e:
        print(f"[!] Failed to fetch URL {base_url}: {e}")
        return

    # Parse the HTML to find all links
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', href=True)
    
    # Get the domain of the base URL to filter out external links
    base_domain = urlparse(base_url).netloc
    found_links_count = 0

    print(f"[*] Found {len(links)} total links. Filtering for internal links on '{base_domain}'...")

    for link in tqdm(links):
        href = link['href']
        
        # Create an absolute URL from the found href
        absolute_url = urljoin(base_url, href)
        
        # Check if the link belongs to the same domain
        if urlparse(absolute_url).netloc == base_domain:
            found_links_count += 1
            # print(f"\n[+] Found internal link: {absolute_url}")
            
            # --- Construct the command ---
            
            # 1. Create a safe filename from the URL path for the output file
            # Example: http://example.com/about/team -> about_team
            path = urlparse(absolute_url).path
            if path in ('', '/'):
                filename = 'index'
            else:
                # Sanitize the path to create a valid filename
                filename = path.strip('/').replace('/', '_').replace('.', '_')
            
            # output_path = os.path.join(output_folder, filename)
            output_path = os.path.join(output_folder, f"{filename}.json")

            # 2. Build the command to be executed
            command = [
                "python",
                TARGET_SCRIPT,
                "--url",
                absolute_url,
                "--output",
                output_path
            ]
            
            # print(f"[*] Executing: {' '.join(command)}")

            # --- Execute the command ---
            try:
                # Using subprocess.run to execute the command
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                # print(f"    - ✔️ Success! Script output saved for {filename}.")
                # To see the script's output, uncomment the following line:
                # print(f"    - STDOUT: {result.stdout.strip()}")
            except FileNotFoundError:
                print(f"[!] Error: The script '{TARGET_SCRIPT}' was not found.")
                print("[!] Please make sure it's in the same directory or in your system's PATH.")
                return # Stop execution if the script is not found
            except subprocess.CalledProcessError as e:
                print(f"    - ❌ Error executing script for {absolute_url}.")
                print(f"    - Return Code: {e.returncode}")
                print(f"    - STDERR: {e.stderr.strip()}")
                
    print(f"\n[*] Finished. Processed {found_links_count} internal links.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script to find all internal links on a webpage and execute a custom script on them."
    )
    parser.add_argument("--url", help="The URL of the webpage to scan.")
    parser.add_argument("--folder_path", help="The path to the folder where the output will be stored.")
    
    args = parser.parse_args()
    
    run_script_on_links(args.url, args.folder_path)