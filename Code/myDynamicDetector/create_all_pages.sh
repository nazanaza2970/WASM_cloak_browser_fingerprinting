#!/bin/bash

# This script runs a two-stage process:
# 1. Runs create_pages.py for an initial set of JSONs.
# 2. Runs create_wasm_pages.py using a second set of JSONs,
#    matching them to the folders created in step 1 based on
#    the filename after the first underscore.

# --- Configuration ---
# Set your paths here
PYTHON_SCRIPT_1="./files/create_pages.py"
PYTHON_SCRIPT_2="./files/create_wasm_pages.py"

JSON_SOURCE_DIR_1="./files/combined_converted/js/" # For create_pages.py
JSON_SOURCE_DIR_2="./files/combined_converted/wasm/" # For create_wasm_pages.py (new jsons)

BASE_TARGET_DIR="./files/pages/" # Base output for BOTH scripts
# ---------------------

# This is an "associative array" (like a dictionary or map)
# It will store mappings like:
# [data1.json] -> /path/to/your/base_output_dir/prefixA_data1
declare -A source_map

# --- Helper Function for Checks ---
check_paths() {
    if [ ! -f "$PYTHON_SCRIPT_1" ]; then
        echo "Error: Script not found: $PYTHON_SCRIPT_1"
        exit 1
    fi
    if [ ! -f "$PYTHON_SCRIPT_2" ]; then
        echo "Error: Script not found: $PYTHON_SCRIPT_2"
        exit 1
    fi
    if [ ! -d "$JSON_SOURCE_DIR_1" ]; then
        echo "Error: JSON source 1 not found: $JSON_SOURCE_DIR_1"
        exit 1
    fi
    if [ ! -d "$JSON_SOURCE_DIR_2" ]; then
        echo "Error: JSON source 2 not found: $JSON_SOURCE_DIR_2"
        exit 1
    fi
    # Ensure the base target directory exists
    mkdir -p "$BASE_TARGET_DIR"
}

# --- STAGE 1: Run create_pages.py ---
run_stage_1() {
    echo "--- STAGE 1: Running $PYTHON_SCRIPT_1 ---"
    
    # *** FIX: Use process substitution < <(...) instead of a pipe | ***
    # This runs the loop in the current shell, so source_map persists.
    # Added -type f to find to only get files.
    # Added -r to read to handle backslashes correctly.
    while read -r json_file; do
        
        # e.g., "prefixA_data1.json"
        filename=$(basename "$json_file")
        
        # e.g., "prefixA_data1"
        name_only="${filename%.json}"
        
        # e.g., "data1.json"
        # This is the key for matching
        match_key="${filename#*_}" 
        
        # e.g., "/path/to/your/base_output_dir/prefixA_data1"
        output_dir_1="$BASE_TARGET_DIR/$name_only"
        
        mkdir -p "$output_dir_1"
        
        echo "Processing 1: $filename  =>  $output_dir_1 (key: $match_key)"
        python3 "$PYTHON_SCRIPT_1" --json "$json_file" --output "$output_dir_1"
        
        # Store the mapping
        source_map["$match_key"]="$output_dir_1"
        
    done < <(find "$JSON_SOURCE_DIR_1" -maxdepth 1 -type f -name "*.json")
    
    echo "--- STAGE 1 Complete. ${#source_map[@]} mappings stored. ---"
}

# --- STAGE 2: Run create_wasm_pages.py ---
run_stage_2() {
    echo "--- STAGE 2: Running $PYTHON_SCRIPT_2 ---"

    # *** FIX: Use process substitution here too for consistency ***
    while read -r new_json_file; do
    
        # e.g., "prefixB_data1.json"
        new_filename=$(basename "$new_json_file")
        
        # e.g., "prefixB_data1"
        new_name_only="${new_filename%.json}"
        
        # e.g., "data1.json"
        match_key="${new_filename#*_}"
        
        # Look up the source directory from Stage 1
        source_dir=${source_map["$match_key"]}
        
        if [ -n "$source_dir" ]; then
            # Match found!
            
            # e.g., "/path/to/your/base_output_dir/prefixB_data1"
            output_dir_2="$BASE_TARGET_DIR/$new_name_only"
            
            mkdir -p "$output_dir_2"
            
            echo "Processing 2: $new_filename (using key: $match_key)"
            echo "  --source_dir: $source_dir"
            echo "  --output_dir: $output_dir_2"
            
            python3 "$PYTHON_SCRIPT_2" \
                --source_dir "$source_dir" \
                --new_json "$new_json_file" \
                --output_dir "$output_dir_2"
        
        else
            # No match found in the map
            echo "Warning: No matching source folder found for $new_filename (key: $match_key)"
        fi
        
    done < <(find "$JSON_SOURCE_DIR_2" -maxdepth 1 -type f -name "*.json")
}

# --- Main Execution ---
check_paths
run_stage_1
run_stage_2
echo "All processes finished."