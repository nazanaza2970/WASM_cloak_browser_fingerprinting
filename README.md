# WASM_cloak_browser_fingerprinting
Artifacts and dataset for the paper "The WASM Cloak: Evaluating Browser Fingerprinting Defenses Under WebAssembly based Obfuscation"


# Datasets
Real-world scripts used for testing and their WASM counterparts are in the folder JS_with_WASM_counterparts
The controlled dataset is in the folder controlled_dataset. Please host the pages in this directory to be able to see the scripts in action.

# Code
The scripts translation_n_pipeline_analysis_v* were used for conversion. All other scripts in the Code directory are either helper scripts or were used in various parts of testing.
The directory myDynamicDetector contains the dynamic analysis tool implemented in our paper. Please move the real-world scripts to myDynamicDetector/files/combined_converted before you start running the scripts in the directory