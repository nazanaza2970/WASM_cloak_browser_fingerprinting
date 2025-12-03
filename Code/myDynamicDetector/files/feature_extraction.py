# filename: process_api_traces_with_ohe.py

import json
import math
import argparse
import logging
from copy import deepcopy

# --- Feature Type Definition ---
# This dictionary mimics the 'feat_type_map.json' from your original script.
# You can extend this map to define which features should be treated as categorical
# and what their possible values are.
FEATURE_TYPE_MAP = {
    'document.getelementsbytagname': {
        'type': 'categorical',
        'items': ['head', 'body', 'div', 'script', 'ins', 'iframe', 'a', 'span', 'canvas']
    },
    'canvasrenderingcontext2d.strokestyle': {
        'type': 'categorical',
        'items': ['gradient', 'color']
    },
    'canvasrenderingcontext2d.fillstyle': {
        'type': 'categorical',
        'items': ['gradient', 'color']
    }
    # Add other categorical features here. For example:
    # 'htmlcanvaselement.getelementsbytagname': {
    #     'type': 'categorical',
    #     'items': [...]
    # }
}


# --- Helper Utilities (re-implemented from utils.py) ---

def is_float(s):
    """Checks if a value can be converted to a float."""
    if s is None:
        return False
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False

def is_arg_float(args, key):
    """Checks if a dictionary value for a given key is a float."""
    return key in args and is_float(args[key])

def is_arg_int(args, key):
    """Checks if a dictionary value for a given key is an integer."""
    if key not in args:
        return False
    val = args[key]
    if isinstance(val, int):
        return True
    if isinstance(val, str):
        return val.isdigit() or (val.startswith('-') and val[1:].isdigit())
    return False

# --- Core Feature Extraction Logic (re-implemented from extract_dataset.py) ---
# Note: The functions from the previous response are included here but are collapsed for brevity.
# (update_onegrams, update_custom_feats_val, update_custom_feats_arg, extract_ground_truth)
def update_onegrams(one_grams, symbol_text):
    """Increments the count for an observed API symbol."""
    one_grams[symbol_text] = one_grams.get(symbol_text, 0) + 1

def update_custom_feats_val(custom_features, symbol_text, api_call):
    """
    Extracts features from the 'value' of an API call.
    """
    api_val = api_call.get('value')
    if api_val is None or api_val == 'null' or str(api_val).strip() == '':
        return
    if symbol_text not in custom_features:
        custom_features[symbol_text] = []
    if len(custom_features[symbol_text]) >= 5:
        return
    if symbol_text == 'window.document.cookie':
        custom_features[symbol_text].append(len(str(api_val).split(';')))
    elif symbol_text in ('canvasrenderingcontext2d.strokestyle', 'canvasrenderingcontext2d.fillstyle', 'canvasrenderingcontext2d.shadowcolor'):
        if '(' in str(api_val):
            custom_features[symbol_text].append('gradient')
        else:
            custom_features[symbol_text].append('color')
    elif symbol_text == 'canvasrenderingcontext2d.filter':
        if '(' in str(api_val):
            custom_features[symbol_text].append(api_val)
        else:
            custom_features[symbol_text].append('')

def update_custom_feats_arg(custom_features, symbol_text, api_call):
    """
    Extracts features from the 'argument' of an API call.
    """
    api_arg_str = api_call.get('argument')
    if api_arg_str is None or api_arg_str == 'null' or api_arg_str.strip() == '':
        return
    try:
        api_args = json.loads(api_arg_str)
        if isinstance(api_args, list):
            api_args = {str(i): v for i, v in enumerate(api_args)}
    except (json.JSONDecodeError, TypeError):
        logging.warning(f"Could not parse argument JSON: {api_arg_str}")
        return
    def init_and_check(feat_name, limit=5):
        if feat_name not in custom_features:
            custom_features[feat_name] = []
        return len(custom_features[feat_name]) < limit
    area_symbols = {'canvasrenderingcontext2d.fillrect', 'canvasrenderingcontext2d.rect', 'canvasrenderingcontext2d.getimagedata', 'canvasrenderingcontext2d.strokerect', 'canvasrenderingcontext2d.clearrect', 'webgl2renderingcontext.viewport', 'webglrenderingcontext.viewport', 'webglrenderingcontext.scissor', 'webgl2renderingcontext.readpixels', 'webglrenderingcontext.readpixels'}
    if symbol_text in area_symbols:
        on_screen_feat = f'{symbol_text}-OnScreen'
        if init_and_check(on_screen_feat):
            feat_val = 1 if is_arg_float(api_args, '0') and is_arg_float(api_args, '1') and float(api_args['0']) >= 0 and float(api_args['1']) >= 0 else 0
            custom_features[on_screen_feat].append(feat_val)
        size_feat = f'{symbol_text}-Size'
        if init_and_check(size_feat):
            size = float(api_args['2']) * float(api_args['3']) if is_arg_float(api_args, '2') and is_arg_float(api_args, '3') else 0
            custom_features[size_feat].append(size)
    elif symbol_text == 'canvasrenderingcontext2d.arc':
        on_screen_feat = f'{symbol_text}-OnScreen'
        if init_and_check(on_screen_feat):
            feat_val = 1 if is_arg_float(api_args, '0') and is_arg_float(api_args, '1') and float(api_args['0']) >= 0 and float(api_args['1']) >= 0 else 0
            custom_features[on_screen_feat].append(feat_val)
        size_feat = f'{symbol_text}-Size'
        if init_and_check(size_feat):
            radius = float(api_args['2'])
            size = math.pi * (radius ** 2) if is_arg_float(api_args, '2') else 0
            custom_features[size_feat].append(size)
    elif symbol_text in ('htmlcanvaselement.getelementsbytagname', 'document.getelementsbytagname', 'audiocontext.createchannelmerger'):
        if init_and_check(symbol_text) and '0' in api_args:
            custom_features[symbol_text].append(api_args['0'])
    # ... Other elif blocks from the original script would continue here ...

def extract_ground_truth(api_calls):
    """Determines fingerprinting behavior based on API call patterns."""
    is_canvas_1, is_canvas_2, is_canvas_3, is_canvas_4 = False, False, False, True
    is_webrtc_1, is_webrtc_2 = False, False
    canvasfont_fonts, canvasfont_measuretext = set(), 0
    is_audio = False
    for api_call in api_calls:
        symbol_text = api_call['symbol'].strip().lower()
        if 'canvasrenderingcontext2d.filltext' in symbol_text or 'canvasrenderingcontext2d.stroketext' in symbol_text: is_canvas_1 = True
        elif 'canvasrenderingcontext2d.fillstyle' in symbol_text or 'canvasrenderingcontext2d.strokestyle' in symbol_text: is_canvas_2 = True
        elif 'htmlcanvaselement.todataurl' in symbol_text: is_canvas_3 = True
        elif symbol_text in ('canvasrenderingcontext2d.save', 'canvasrenderingcontext2d.restore', 'canvasrenderingcontext2d.addeventlistener'): is_canvas_4 = False
        elif symbol_text in ('rtcpeerconnection.createdatachannel', 'rtcpeerconnection.createoffer'): is_webrtc_1 = True
        elif symbol_text in ('rtcpeerconnection.onicecandidate', 'rtcpeerconnection.localdescription'): is_webrtc_2 = True
        elif 'canvasrenderingcontext2d.font' in symbol_text: canvasfont_fonts.add(api_call.get('value'))
        elif 'canvasrenderingcontext2d.measuretext' in symbol_text: canvasfont_measuretext += 1
        elif symbol_text in ('offlineaudiocontext.createoscillator', 'offlineaudiocontext.createdynamicscompressor', 'offlineaudiocontext.destination', 'offlineaudiocontext.startrendering', 'offlineaudiocontext.oncomplete'): is_audio = True
    is_canvas = is_canvas_1 and is_canvas_2 and is_canvas_3 and is_canvas_4
    is_webrtc = is_webrtc_1 and is_webrtc_2
    is_canvasfont = len(canvasfont_fonts) > 20 and canvasfont_measuretext > 20
    is_fp = is_canvas or is_webrtc or is_canvasfont or is_audio
    return {'is_canvas': is_canvas, 'is_webrtc': is_webrtc, 'is_canvasfont': is_canvasfont, 'is_audio': is_audio, 'is_fingerprinting': is_fp}

# --- New One-Hot Encoding Functionality ---

def one_hot_encode_features(script_features, feature_map):
    """
    Applies one-hot encoding to categorical features based on the provided map.

    Args:
        script_features (dict): The dictionary of extracted raw features for a script.
        feature_map (dict): A map defining categorical features and their possible values.

    Returns:
        dict: A new dictionary containing the one-hot encoded features.
    """
    encoded_features = {}
    
    # Combine features from 'value' and 'argument' for easier processing
    all_custom_features = {
        **script_features.get("custom_features_from_value", {}),
        **script_features.get("custom_features_from_argument", {})
    }

    for feat_name, observed_values in all_custom_features.items():
        if feat_name in feature_map and feature_map[feat_name]['type'] == 'categorical':
            
            # This is a categorical feature, so we one-hot encode it
            category_items = feature_map[feat_name]['items']
            num_categories = len(category_items)
            
            encoded_vectors = []
            for value in observed_values:
                # Create a default vector for 'unknown' values
                one_hot_vector = [0] * num_categories
                
                try:
                    # Find the index of the observed value in the predefined list
                    idx = category_items.index(value)
                    one_hot_vector[idx] = 1
                except ValueError:
                    # The value was not found in our list of known categories.
                    # We can either ignore it or add a separate 'unknown' category.
                    # For now, we log it and append a vector of zeros.
                    logging.warning(f"Value '{value}' for feature '{feat_name}' not in defined categories. Using zero-vector.")
                
                encoded_vectors.append(one_hot_vector)
            
            encoded_features[feat_name] = encoded_vectors
        else:
            # This is a numerical or raw feature, so we keep it as is
            encoded_features[feat_name] = observed_values
            
    return encoded_features


# --- Main Processing Logic ---

def process_traces(file_path):
    """
    Main function to process the API log file, extract features, and apply one-hot encoding.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {file_path}")
        return

    all_script_results = {}

    for script_hash, script_data in data.items():
        api_calls = script_data.get('information', [])
        
        one_grams = {}
        custom_features_val = {}
        custom_features_arg = {}

        for api_call in api_calls:
            if 'symbol' not in api_call: continue
            symbol_text = api_call['symbol'].strip().lower()
            update_onegrams(one_grams, symbol_text)
            update_custom_feats_val(custom_features_val, symbol_text, api_call)
            update_custom_feats_arg(custom_features_arg, symbol_text, api_call)

        ground_truth = extract_ground_truth(api_calls)
        
        # Structure the raw extracted features
        raw_features = {
            "content_hash": script_data.get("content_hash", script_hash),
            "one_grams": one_grams,
            "custom_features_from_value": custom_features_val,
            "custom_features_from_argument": custom_features_arg,
        }
        
        # Apply one-hot encoding
        one_hot_encoded = one_hot_encode_features(raw_features, FEATURE_TYPE_MAP)

        all_script_results[script_hash] = {
            "raw_features": raw_features,
            "one_hot_encoded_features": one_hot_encoded,
            "ground_truth_labels": ground_truth
        }

    # Print the result as a nicely formatted JSON
    print(json.dumps(all_script_results, indent=4))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description="Extract features from Playwright API traces and apply one-hot encoding.")
    parser.add_argument('file_path', type=str, help='Path to the input JSON file (e.g., 178_html.json).')
    args = parser.parse_args()
    process_traces(args.file_path)