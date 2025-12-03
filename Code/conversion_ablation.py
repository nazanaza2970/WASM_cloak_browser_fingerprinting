import esprima  # for JavaScript AST
import subprocess  # to run AssemblyScript compiler
import tempfile
import asyncio, concurrent.futures, multiprocessing as mp
import os
import glob
import escodegen
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Dict, Any, List, Tuple, Optional
# from typing import List, Dict, Optional
import re
from textwrap import dedent
import pickle
import random
from collections import defaultdict

from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright

import time
import csv


# In[4]:


model_name = "Qwen/Qwen2.5-Coder-14B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "write a quick sort algorithm."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)


# In[5]:


def convert_js_to_asm_qwen(code,user_msg):
    messages = [
        {"role": "user", "content": user_msg+code},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def extract_assemblyscript_code(text):
    pattern = r"<\|im_start\|>assistant(.*?)<\|im_end\|>"
    pattern_2 = r'```assemblyscript(.*?)```'
    pattern_3 = r'```typescript(.*?)```'

    if '`assemblyscript' in text:
        match = re.search(pattern_2, text, re.DOTALL)
    elif '`typescript' in text:
        match = re.search(pattern_3, text, re.DOTALL)
    else:
        match = re.search(pattern, text, re.DOTALL)  # re.DOTALL allows the dot (.) to match newlines
    if match:
        return match.group(1).strip()
    else:
        return text


# In[6]:


#old utils

def line_col_to_index(code, line, column):
    # Split the code into lines
    lines = code.splitlines(True)  # keep line breaks
    # Ensure line and column are within bounds
    if line < 1 or line > len(lines):
        return -1
    # The index is the sum of all previous lines' lengths plus the column number in the current line
    return sum(len(lines[i]) for i in range(line - 1)) + (column)

def compile_asm(asm_code):
    """
    Compile AssemblyScript code to WebAssembly binary
    """
    # Write AssemblyScript code to a temporary .ts file
    with tempfile.NamedTemporaryFile(suffix=".ts", mode='w', delete=False) as asm_file:
        asm_file.write(asm_code)
        asm_filename = asm_file.name

    # Create output filename
    output_filename = asm_filename.replace(".ts", ".wasm")
    
    try:
        # Get the npm binary path
        npm_path = subprocess.check_output(['which', 'npm'], text=True).strip()
        node_path = os.path.dirname(os.path.dirname(npm_path))
        
        # Add node_modules to PATH
        env = os.environ.copy()
        env['PATH'] = f"{node_path}/bin:{env['PATH']}"
        
        # Run AssemblyScript compiler
        result = subprocess.run(
            ["asc", asm_filename, "-o", output_filename],
            capture_output=True,
            text=True,
            env=env,
            check=False  # Don't raise exception immediately
        )
        
        # Check for errors
        if result.returncode != 0:
            raise RuntimeError(f"AssemblyScript compilation failed:\n{result.stderr}")
            
        # Read and return the WebAssembly binary
        with open(output_filename, "rb") as wasm_file:
            return wasm_file.read()
            
    except FileNotFoundError:
        raise RuntimeError("AssemblyScript compiler (asc) not found. Please install it with: npm install -g assemblyscript")
    except Exception as e:
        raise RuntimeError(f"Compilation error: {str(e)}")
    finally:
        # Cleanup temporary files
        if os.path.exists(asm_filename):
            os.remove(asm_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)

def run_js(js_code):
    try:
        result = subprocess.run(["node", "-e", js_code], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("JavaScript executed successfully!")
            return True
        else:
            print(f"JavaScript execution error: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error running JavaScript: {e}")
        return False


# In[7]:


#new utils

def get_mod_js(js_code, chunks):
    # Sort chunks by start_ind in ascending order
    chunks = sorted(chunks, key=lambda chunk: chunk['start_ind'])
    
    # Remove overlapping chunks
    non_overlapping_chunks = []
    last_end = -1
    
    for chunk in chunks:
        if chunk['start_ind'] >= last_end:
            non_overlapping_chunks.append(chunk)
            last_end = chunk['end_ind']
    
    chunked = []
    last_e = 0
    total_converted = 0
    for chunk in non_overlapping_chunks:
        chunked.append(js_code[last_e:chunk['start_ind']])
        chunked.append(chunk['code'])
        last_e = chunk['end_ind']
        total_converted+=(chunk['end_ind']-chunk['start_ind'])
    
    chunked.append(js_code[last_e:])
    mod_js = "".join(chunked)

    total_converted = total_converted/len(js_code) if len(js_code) > 0 else 0
    return mod_js,total_converted

def get_mod_js_old(js_code, chunks):
    # Sort chunks by start_ind in ascending order
    chunks = sorted(chunks, key=lambda chunk: chunk['start_ind'])
    
    chunked = []
    last_e = 0
    
    for chunk in chunks:
        chunked.append(js_code[last_e:chunk['start_ind']])
        chunked.append(chunk['code'])
        last_e = chunk['end_ind']
    
    chunked.append(js_code[last_e:])
    mod_js = "".join(chunked)
    return mod_js

def get_binding_code_general(var_name,start_ind):

    binding_code = f"let {var_name} = instance_lit.exports.{var_name}_{start_ind}.value"
    return binding_code

def get_binding_code_string_literal(var_name,start_ind):

    binding_code = f"let {var_name} = getString(instance_lit.exports.{var_name}_{start_ind})"
    return binding_code

def get_import_object(js_dict=None):# called fn with arguments
    if js_dict:
        import_object = f"""{{ env: {{
    memory: new WebAssembly.Memory({{ initial: 64 }}), // Adjust memory size as needed
    __alloc: function (size) {{
      // Allocate memory and return a pointer (adjust this function as needed for your runtime)
      return memory_lit.grow(Math.ceil((size + memory_lit.buffer.byteLength) / 65536)) || 0;
    }},
    store: function (ptr, value) {{
      // Store a value at a memory pointer
      let memoryView = new Int32Array(memory_lit.buffer);
      memoryView[ptr / 4] = value; // Assuming 32-bit integers
    }},
    abort: function () {{
      console.error("Abort called");
      throw new Error("Abort called");
    }},
  }},
            js: {js_dict} // add dictionary of functions with positions here
        }};
        """
    else:
        import_object = f""" {{env: {{
    memory: new WebAssembly.Memory({{ initial: 64 }}), // Adjust memory size as needed
    __alloc: function (size) {{
      // Allocate memory and return a pointer (adjust this function as needed for your runtime)
      return memory_lit.grow(Math.ceil((size + memory_lit.buffer.byteLength) / 65536)) || 0;
    }},
    store: function (ptr, value) {{
      // Store a value at a memory pointer
      let memoryView = new Int32Array(memory_lit.buffer);
      memoryView[ptr / 4] = value; // Assuming 32-bit integers
    }},
    abort: function () {{
      console.error("Abort called");
      throw new Error("Abort called");
    }},
  }},
 }};       """

    return import_object

def get_wasm_loading_code(asm_code,mod_js, has_string=False,js_dict=None):
    wasm_bytes = compile_asm(asm_code)
    import_object = get_import_object(js_dict)
    wasm_loader = f"""
let loadWasm = (async function (){{
    let wasmBinary_lit = new Uint8Array({list(wasm_bytes)});
    const imports = {import_object}
    let obj_lit = await WebAssembly.instantiate(wasmBinary_lit,imports); //need to add proper env for func, if-else, loops
    let instance_lit = obj_lit.instance;
    let memory_lit = instance_lit.exports.memory;

    return [instance_lit,memory_lit];
    }})();


    loadWasm.then((results)=>{{
    let [instance_lit, memory_lit] = results;
    function getString(ptr) {{
      let len = new Uint32Array(memory_lit.buffer, ptr - 4, 1)[0];
      let strBuffer = new Uint8Array(memory_lit.buffer, ptr, len);
      let str = '';
      for (let i = 0; i < len; i++) {{
        let charCode = strBuffer[i];
        if (charCode !== 0) {{  // Skip null characters (if any)
          str += String.fromCharCode(charCode);}}
      }}
      return str;}}

    {mod_js}
    }});
    """
    return wasm_loader


# In[8]:


# rule-1 : replace all literal    

def replace_literals_recursive(js_code, ast_g, node=None, chunks_to_merge=None, asm_code=None):
    if chunks_to_merge is None:
        chunks_to_merge = []
    if asm_code is None:
        asm_code = []
    if node is None:
        node = ast_g                                 # root on first entry

    def process_declaration(declarator, start_pos, end_pos):
        start_ind = line_col_to_index(js_code, start_pos.line, start_pos.column)
        end_ind   = line_col_to_index(js_code, end_pos.line,   end_pos.column)

        if declarator.init and declarator.init.type == "Literal":
            var_name = declarator.id.name

            if isinstance(declarator.init.value, str):
                value = declarator.init.value
                asm_code.append(f'export let {var_name}_{start_ind}: string = "{value}";\n')
                binding_code = get_binding_code_string_literal(var_name, start_ind)

            elif isinstance(declarator.init.value, bool):
                value = 1 if declarator.init.value else 0
                asm_code.append(f'export let {var_name}_{start_ind}: i32 = {value};\n')
                binding_code = get_binding_code_general(var_name, start_ind)

            elif isinstance(declarator.init.value, int):
                value = declarator.init.value
                asm_code.append(f'export let {var_name}_{start_ind}: i32 = {value};\n')
                binding_code = get_binding_code_general(var_name, start_ind)

            elif isinstance(declarator.init.value, float):
                value = declarator.init.value
                asm_code.append(f'export let {var_name}_{start_ind}: f64 = {value};\n')
                binding_code = get_binding_code_general(var_name, start_ind)

            else:
                return  # unsupported type

            chunks_to_merge.append({
                "code":      binding_code,
                "start_ind": start_ind,
                "end_ind":   end_ind
            })

    # ----- Traversal --------------------------------------------------------
    if hasattr(node, "body") and isinstance(node.body, list):
        for child in node.body:
            # FIX: respect positional order (js_code, ast_g, node, chunks_to_merge, asm_code)
            replace_literals_recursive(js_code, ast_g, child, chunks_to_merge, asm_code)

    elif getattr(node, "type", None) == "VariableDeclaration":
        for declarator in node.declarations:
            start_pos = node.loc.start
            end_pos   = declarator.loc.end
            process_declaration(declarator, start_pos, end_pos)

    elif getattr(node, "type", None) == "ForStatement":
        for part in ("init", "test", "update"):
            if hasattr(node, part) and getattr(node, part) and getattr(node, part).type == "Literal":
                continue  # skip literals inside for-loop header
        replace_literals_recursive(js_code, ast_g, node.body, chunks_to_merge, asm_code)

    elif hasattr(node, "expression") and node.expression:
        replace_literals_recursive(js_code, ast_g, node.expression, chunks_to_merge, asm_code)

    elif hasattr(node, "__dict__"):
        for key, value in node.__dict__.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) or hasattr(item, "__dict__"):
                        replace_literals_recursive(js_code, ast_g, item, chunks_to_merge, asm_code)
            elif isinstance(value, dict) or hasattr(value, "__dict__"):
                replace_literals_recursive(js_code, ast_g, value, chunks_to_merge, asm_code)

    return chunks_to_merge, asm_code, []


# In[9]:


# rule 2 : function obfuscation

# Function IDs
FUNCTION_IDS = {
    "eval": 0,
    "escape": 1,
    "atob": 2,
    "btoa": 3,
    # "WScript": 4,
    "unescape": 5,
    "Function": 6,
    "ActiveXObject": 7,
}

def extract_arguments_r2(arguments):
    """
    Extract argument representations from the AST nodes.
    """
    extracted_args = []
    for arg in arguments:
        if arg.get("type") == "Literal":
            extracted_args.append(arg.get("value"))  # For literals, add the value
        elif arg.get("type") == "Identifier":
            extracted_args.append(arg.get("name"))  # For identifiers, add the name
        else:
            extracted_args.append("complex_expression")  # Placeholder for complex expressions
    return extracted_args

def find_functions_with_positions_r2(ast, functions_to_find):
    """
    Recursively traverse the AST to find specified functions, their IDs, positions, and arguments.
    """
    results = []

    def traverse(node):
        if isinstance(node, dict):
            # Check for CallExpression nodes
            if node.get("type") == "CallExpression":
                callee = node.get("callee")
                if callee:
                    # Check for simple function calls
                    if callee.get("type") == "Identifier" and callee.get("name") in functions_to_find:
                        results.append({
                            "function": callee["name"],
                            "id": FUNCTION_IDS[callee["name"]],
                            "start": node["loc"]["start"],
                            "end": node["loc"]["end"],
                            "arguments": extract_arguments_r2(node.get("arguments", [])),
                        })
                    # Check for WScript-style calls (MemberExpression)
                    elif callee.get("type") == "MemberExpression":
                        obj_name = callee["object"].get("name")
                        if obj_name in functions_to_find:
                            results.append({
                                "function": f"{obj_name}.{callee['property']['name']}",
                                "id": FUNCTION_IDS[obj_name],
                                "start": node["loc"]["start"],
                                "end": node["loc"]["end"],
                                "arguments": extract_arguments_r2(node.get("arguments", [])),
                            })

            # Check for NewExpression nodes
            elif node.get("type") == "NewExpression" and node.get("callee", {}).get("type") == "Identifier":
                func_name = node["callee"]["name"]
                if func_name in functions_to_find:
                    results.append({
                        "function": func_name,
                        "id": FUNCTION_IDS[func_name],
                        "start": node["loc"]["start"],
                        "end": node["loc"]["end"],
                        "arguments": extract_arguments_r2(node.get("arguments", [])),
                    })

            # Recurse into child nodes
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    traverse(value)
        elif isinstance(node, list):
            for item in node:
                traverse(item)

    traverse(ast)
    return results

def process_js_code_r2(js_code,FUNCTION_IDS,ast_g):
    """
    Parse the JavaScript code and find specified functions with positions.
    """
    try:
        ast = esprima.parseScript(js_code, tolerant=True, loc=True)
        return find_functions_with_positions_r2(ast.toDict(), FUNCTION_IDS.keys())
    except Exception as e:
        print(f"Error parsing JavaScript code: {e}")
        return []


def obfuscate_functions(js_code,ast_g):
    FUNCTION_IDS = {
        "eval": 0,
        "escape": 1,
        "atob": 2,
        "btoa": 3,
        # "WScript": 4,
        "unescape": 5,
        "Function": 6,
        "ActiveXObject": 7,
    }
    def get_binding_code_f(func,start_ind):
        binding_code=f"""
        (function(){{
        let pointer_{func['function']}_{start_ind} = instance_lit.exports.{func['function']}_{start_ind};
        const globalObject_{start_ind} = typeof window !== 'undefined' ? window : global;
        globalObject_{start_ind}[getString(pointer_{func['function']}_{start_ind})]({func['arguments'][0]});
        }})();
        """

        return binding_code
        
    # Process the code
    found_functions = process_js_code_r2(js_code,FUNCTION_IDS,ast_g)
    
    chunks_to_merge = []
    asm_code = []
    for func in found_functions:
        start_ind = line_col_to_index(js_code, func['start']['line'], func['start']['column'])
        end_ind = line_col_to_index(js_code, func['end']['line'], func['end']['column'])

        asm_code.append(f"""export let {func['function']}_{start_ind}: string = "{func['function']}";\n""")
        binding_code = get_binding_code_f(func,start_ind)

        if js_code[start_ind:end_ind].startswith("new "):
            continue

        chunks_to_merge.append({
            'code': binding_code,
            'start_ind': start_ind,
            'end_ind': end_ind
        })
        
    return chunks_to_merge, asm_code, []


# In[10]:


# rule 3a: int arrays

def convert_int_array_to_assemblyscript(arr,start_pos):
    """Convert numeric arrays to AssemblyScript code with memory pointer generation"""
    assemblyscript_code = ""
    
    arr_name = arr.id.name
    arr_values = [e.value for e in arr.init.elements]
    
    # Generate AssemblyScript array initialization code with WebAssembly memory
    assemblyscript_code += f"export function get_{arr_name}_{start_pos}Pointer(): i32 {{\n"
    assemblyscript_code += f"  const {arr_name} = new Array<i32>({len(arr_values)});\n"
    
    # Assign values to the array in AssemblyScript
    for i, value in enumerate(arr_values):
        assemblyscript_code += f"  {arr_name}[{i}] = {value};\n"
    
    # Create a memory array and copy the values to WebAssembly memory
    assemblyscript_code += f"  const ptr = __alloc({len(arr_values)} * sizeof<i32>());\n"
    assemblyscript_code += f"  for (let i = 0; i < {arr_name}.length; i++) {{\n"
    assemblyscript_code += f"    store<i32>(ptr + i * sizeof<i32>(), {arr_name}[i]);\n"
    assemblyscript_code += f"  }}\n"
    
    # Return the pointer to the start of the array in memory
    assemblyscript_code += f"  return ptr;\n"
    assemblyscript_code += f"}}\n\n"
    
    return assemblyscript_code,arr_name,len(arr_values)

def get_binding_code_int_arrays(arr_name,arrayLength,start_pos):
    binding_code = f"""
    const createArray_{arr_name} = instance_lit.exports.get_{arr_name}_{start_pos}Pointer;
    const ptr_{arr_name} = createArray_{arr_name}();
    const {arr_name} = new Int32Array(instance_lit.exports.memory.buffer, ptr_{arr_name}, {arrayLength});
    """

    return binding_code

def replace_int_arrays(js_code,ast_g):
    ast = ast_g
    # ast = esprima.parseScript(js_code, loc=True)
    # ar_count = 0
    chunks_to_merge = []
    asm_code = []

    def traverse_node(node):
        """Recursively traverse the AST to find array literals with only integers."""
        if isinstance(node, esprima.nodes.VariableDeclaration):
            for declarator in node.declarations:
                start_pos = node.loc.start
                end_pos = declarator.loc.end
                start_ind = line_col_to_index(js_code, start_pos.line, start_pos.column)
                end_ind = line_col_to_index(js_code, end_pos.line, end_pos.column)

                if isinstance(declarator.init, esprima.nodes.ArrayExpression):
                    # Check if the array contains only int values
                    if all(isinstance(el.value, (int)) for el in declarator.init.elements):
                        # ar_count+=1
                        assemblyscript_code, arr_name, arrayLength = convert_int_array_to_assemblyscript(declarator,start_ind)
                        asm_code.append(assemblyscript_code)
                        binding_code = get_binding_code_int_arrays(arr_name, arrayLength,start_ind)

                        # keep chunks to merge
                        chunks_to_merge.append({
                            'code': binding_code,
                            'start_ind': start_ind,
                            'end_ind': end_ind
                        })

                        # reset asm code template
                        assemblyscript_code = ""

        # Now traverse all children recursively
        for key, value in node.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, esprima.nodes.Node):
                        traverse_node(item)
            elif isinstance(value, esprima.nodes.Node):
                traverse_node(value)

    # Start the recursive traversal
    traverse_node(ast)
    return chunks_to_merge, asm_code, []


# In[11]:


# rule 3b: float arrays

def convert_float_array_to_assemblyscript(arr,start_pos):
    """Convert numeric arrays to AssemblyScript code with memory pointer generation"""
    assemblyscript_code = ""
    
    arr_name = arr.id.name
    arr_values = [e.value for e in arr.init.elements]
    
    # Generate AssemblyScript array initialization code with WebAssembly memory
    assemblyscript_code += f"export function get_{arr_name}_{start_pos}Pointer(): i32 {{\n"
    assemblyscript_code += f"  const {arr_name} = new Array<f32>({len(arr_values)});\n"
    
    # Assign values to the array in AssemblyScript
    for i, value in enumerate(arr_values):
        assemblyscript_code += f"  {arr_name}[{i}] = {value};\n"
    
    # Create a memory array and copy the values to WebAssembly memory
    assemblyscript_code += f"  const ptr = __alloc({len(arr_values)} * sizeof<f32>());\n"
    assemblyscript_code += f"  for (let i = 0; i < {arr_name}.length; i++) {{\n"
    assemblyscript_code += f"    store<f32>(ptr + i * sizeof<f32>(), {arr_name}[i]);\n"
    assemblyscript_code += f"  }}\n"
    
    # Return the pointer to the start of the array in memory
    assemblyscript_code += f"  return ptr;\n"
    assemblyscript_code += f"}}\n\n"
    
    return assemblyscript_code,arr_name,len(arr_values)

def get_binding_code_float_arrays(arr_name,arrayLength,start_pos):
    binding_code = f"""
    const createArray_{arr_name} = instance_lit.exports.get_{arr_name}_{start_pos}Pointer;
    const ptr_{arr_name} = createArray_{arr_name}();
    const {arr_name} = new Float32Array(instance_lit.exports.memory.buffer, ptr_{arr_name}, {arrayLength});
    """

    return binding_code

def replace_float_arrays(js_code,ast_g):
    ast = ast_g
    # ast = esprima.parseScript(js_code, loc=True)
    # ar_count = 0
    chunks_to_merge = []
    asm_code = []

    def traverse_node(node):
        """Recursively traverse the AST to find array literals with only float values."""
        if isinstance(node, esprima.nodes.VariableDeclaration):
            for declarator in node.declarations:
                start_pos = node.loc.start
                end_pos = declarator.loc.end
                start_ind = line_col_to_index(js_code, start_pos.line, start_pos.column)
                end_ind = line_col_to_index(js_code, end_pos.line, end_pos.column)

                if isinstance(declarator.init, esprima.nodes.ArrayExpression):
                    # Check if the array contains only float values
                    if all(isinstance(el.value, (float)) for el in declarator.init.elements):
                        # ar_count+=1
                        assemblyscript_code, arr_name, arrayLength = convert_float_array_to_assemblyscript(declarator,start_ind)
                        asm_code.append(assemblyscript_code)
                        binding_code = get_binding_code_float_arrays(arr_name, arrayLength,start_ind)

                        # keep chunks to merge
                        chunks_to_merge.append({
                            'code': binding_code,
                            'start_ind': start_ind,
                            'end_ind': end_ind
                        })

                        # reset asm code template
                        assemblyscript_code = ""

        # Now traverse all children recursively
        for key, value in node.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, esprima.nodes.Node):
                        traverse_node(item)
            elif isinstance(value, esprima.nodes.Node):
                traverse_node(value)

    # Start the recursive traversal
    traverse_node(ast)
    return chunks_to_merge, asm_code, []


# In[12]:


# rule 4 : if-else

def get_if_else_locations(js_code,node, positions):
    keywords = {"break", "continue", "return", "yield", "throw"}
    if getattr(node, 'type', None) == 'IfStatement':
        
        def contains_keywords(code_block):
            return any(keyword in code_block for keyword in keywords)

        if node.test:
            test = escodegen.generate(node.test) if node.test else None
        if node.consequent:
            conseq=escodegen.generate(node.consequent).strip('{').strip('}').strip('\n').strip(' ').strip(';') if node.consequent else None
        if node.alternate:
            alternate = escodegen.generate(node.alternate).strip('{').strip('}').strip('\n').strip(' ').strip(';') if node.alternate else None
        else:
            alternate="null"

        if contains_keywords(conseq) or contains_keywords(alternate):
            return

        # Add the location of the IfStatement
        if hasattr(node, 'loc'):
            pos = node.loc
            positions.append(
                {
                    'pos':(line_col_to_index(js_code,pos.start.line,pos.start.column),line_col_to_index(js_code,pos.end.line,pos.end.column)),
                    'test':test,
                    'conseq':conseq,
                    'alternate':alternate,
                }
            )
        
        # Recursively check the 'consequent' and 'alternate' parts
        if hasattr(node, 'consequent'):
            get_if_else_locations(js_code,node.consequent, positions)
        
        if hasattr(node, 'alternate') and node.alternate is not None:
            get_if_else_locations(js_code,node.alternate, positions)

    # If the node has children, recursively traverse them
    for key in dir(node):
        value = getattr(node, key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) or hasattr(item, 'type'):  # Check for dict or object
                    get_if_else_locations(js_code,item, positions)
        elif isinstance(value, dict) or hasattr(value, 'type'):
            get_if_else_locations(js_code,value, positions)

def find_if_else(js_code,ast_g):
    positions = []
    ast = ast_g
    # ast = esprima.parseScript(js_code,loc=True)
    get_if_else_locations(js_code, ast, positions)
    return positions


def replace_if_else(js_code,ast_g):
    
    def get_asm_if(start_pos):
        asm_if = f"""
        // Define imports for JS functions
        @external("js", "$imp1_{start_pos}")
        declare function $imp1_{start_pos}(): void;
        
        @external("js", "$imp2_{start_pos}")
        declare function $imp2_{start_pos}(): void;
        
        // Exported function to evaluate condition and call the appropriate function
        export function $if_else_{start_pos}(condition: i32): void {{
          if (condition == 1) {{
            $imp1_{start_pos}(); // Call $imp1 if condition is true
          }} else {{
            $imp2_{start_pos}(); // Call $imp2 if condition is false
          }}
        }}
        """
        return asm_if

    def get_js_dict_obj_while(position):
        # params = ", ".join(position['variables']) if position['variables'] else "" ##
        obj = f"""
            $imp1_{position['pos'][0]}: () => {{{position['conseq']}}},
            $imp2_{position['pos'][0]}: () => {{{position['alternate']}}},
            """
        return obj

    def get_binding_code_if(position):
        binding_code = f"""
        let wasmTestCondition_{position['pos'][0]} = {position['test']} ? 1 : 0; // Compute the condition in JS
        instance_lit.exports.$if_else_{position['pos'][0]}(wasmTestCondition_{position['pos'][0]});
        """
        return binding_code

    if_else_positions = find_if_else(js_code,ast_g)
    
    chunks_to_merge = []
    asm_code = []
    js_obj_dict_list = []
    for item in if_else_positions:
        asm_code.append(get_asm_if(item['pos'][0]))
        js_obj_dict_list.append(get_js_dict_obj_while(item))
        binding_code = get_binding_code_if(item)
        chunks_to_merge.append({
        'code':binding_code,
        'start_ind': item['pos'][0],
        'end_ind':item['pos'][1]
        })

    return chunks_to_merge,asm_code, js_obj_dict_list

# find_if_else(js_code)


# In[13]:


# rule 5 : for loop
def find_for_loops(js_code,node,positions):
    if getattr(node, 'type', None) == 'ForStatement':

        # Add the location of the ForStatement
        if hasattr(node, 'loc') and len(node.init.declarations)==1:
            pos = node.loc
            condition = node.test
            body = node.body
            increment = node.update
            idx = node.init.declarations[0].id.name
            value_of_idx = escodegen.generate(node.init.declarations[0].init) if node.init.declarations[0].init else None ##
            positions.append(
                {
                    'pos':(line_col_to_index(js_code,pos.start.line,pos.start.column),line_col_to_index(js_code,pos.end.line,pos.end.column)),
                    'condition':escodegen.generate(condition) if condition else None,
                    'body':escodegen.generate(body).strip('{').strip('}').strip('\n').strip(' ') if body else None,
                    'increment':escodegen.generate(increment) if increment else None,
                    'idx' : idx,
                    'value_of_idx':value_of_idx,
                }
            )

    # If the node has children, recursively traverse them
    for key in dir(node):
        value = getattr(node, key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) or hasattr(item, 'type'):  # Check for dict or object
                    find_for_loops(js_code,item,positions)
        elif isinstance(value, dict) or hasattr(value, 'type'):
            find_for_loops(js_code,value,positions)




def replace_for_loops(js_code,ast_g):
    ast = ast_g
    # ast = esprima.parseScript(js_code, loc=True)
    for_positions = []
    find_for_loops(js_code,ast,for_positions)
    # ast
    # positions
    def get_asm_for(position):
        asm_code_for = f"""          
            @external("js", "body_{position['pos'][0]}")
            declare function body_{position['pos'][0]}(): void;
            
            export function for_{position['pos'][0]}(): void {{
            let {position['idx']}: i32={position['value_of_idx']};
              while ({position['condition']}) {{
                body_{position['pos'][0]}();
                {position['increment']};
              }}
            }}
        """
        return asm_code_for

    def get_binding_code_for(position):
        binding_code = f"instance_lit.exports.for_{position['pos'][0]}();"

        return binding_code

    def get_js_dict_obj(position):
        obj = f"body_{position['pos'][0]}: () => {{{position['body']}}},"

        return obj
    
    chunks_to_merge = []
    asm_code = []
    js_dict_list = []
    for item in for_positions:
        asm_code.append(get_asm_for(item))
        js_dict_list.append(get_js_dict_obj(item))
        binding_code = get_binding_code_for(item)
        chunks_to_merge.append({
        'code':binding_code,
        'start_ind': item['pos'][0],
        'end_ind':item['pos'][1]
        })
    
    return chunks_to_merge, asm_code, js_dict_list


# In[14]:


# rule 6 : while loop

def extract_variables_from_expression(expression):
    """Extracts variable names from a JavaScript expression."""
    variables = set()

    def traverse(node):
        if node.type == "Identifier":
            variables.add(node.name)
        for key in dir(node):
            value = getattr(node, key, None)
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, "type"):
                        traverse(item)
            elif hasattr(value, "type"):
                traverse(value)

    traverse(expression)
    return list(variables)

def find_while_loops(js_code,node,positions):
    if getattr(node, 'type', None) == "WhileStatement":
        if hasattr(node, 'loc'):
            pos = node.loc
            condition = node.test
            body = node.body
            variables = extract_variables_from_expression(condition) if condition else [] ##

            positions.append(
                {
                    'pos':(line_col_to_index(js_code,pos.start.line,pos.start.column),line_col_to_index(js_code,pos.end.line,pos.end.column)),
                    'condition':escodegen.generate(condition) if condition else None,
                    'body':escodegen.generate(body).strip('{').strip('}').strip('\n').strip(' ') if body else None,
                    'variables':variables##
                }
            )

    # If the node has children, recursively traverse them
    for key in dir(node):
        value = getattr(node, key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) or hasattr(item, 'type'):  # Check for dict or object
                    find_while_loops(js_code,item,positions)
        elif isinstance(value, dict) or hasattr(value, 'type'):
            find_while_loops(js_code,value,positions)


def replace_while_loops(js_code,ast_g):
    while_positions = []    
    ast = ast_g
    # ast = esprima.parseScript(js_code,loc=True)
    find_while_loops(js_code,ast,while_positions)
    
    def get_asm_while(position):
        params_decl = ", ".join([f"{var}: i32" for var in position['variables']])  ##
    
        # Create parameter passing (without types)
        params_pass = ", ".join(position['variables']) ##
        
        asm_code = f"""
        @external("js", "cond_{position['pos'][0]}")
        declare function cond_{position['pos'][0]}({params_decl}): i32; //##
        
        @external("js", "body_{position['pos'][0]}")
        declare function body_{position['pos'][0]}({params_decl}): void; //##
        
        export function f_{position['pos'][0]}({params_decl}): void {{ //##
          while (true) {{
            if (cond_{position['pos'][0]}({params_pass}) == 0) {{
              break;
            }}
            body_{position['pos'][0]}({params_pass});
          }}
        }}
    """    
        return asm_code
    
    def get_js_dict_obj_while(position):
        params = ", ".join(position['variables']) if position['variables'] else "" ##
        obj = f"""
            cond_{position['pos'][0]}: ({params}) => {position['condition']}?1:0,//##
            body_{position['pos'][0]}: ({params}) => {{ //##
                {position['body']}
            }},
            """
        return obj
    
    def get_binding_code_while(position):     
        params = ", ".join(position['variables']) if position['variables'] else "" ##
        binding_code = f"instance_lit.exports.f_{position['pos'][0]}();"
        # binding_code = f"instance_lit.exports.f_{position['pos'][0]}({params});" # correct code, commented to stop infinite loop
        return binding_code
    
    chunks_to_merge = []
    asm_code = []
    js_dict_list = []
    for item in while_positions:
        asm_code.append(get_asm_while(item))
        js_dict_list.append(get_js_dict_obj_while(item))
        binding_code = get_binding_code_while(item)
        chunks_to_merge.append({
        'code':binding_code,
        'start_ind': item['pos'][0],
        'end_ind':item['pos'][1]
        })

    return chunks_to_merge,asm_code,js_dict_list

# replace_while_loops(js_code)


# In[15]:


# rule 7a : function calls without returns

def is_function_call_without_return(node, parent):
    """
    Determines if the node is a function call with no return value.
    - The node must be a CallExpression.
    - It should not be part of an AssignmentExpression or another expression using its return value.
    """
    if node.type != "CallExpression":
        return False
    
    # Check if the parent node uses the return value
    if parent:
        # Parent types that imply return value usage
        if parent.type in {"AssignmentExpression", "VariableDeclarator", "ReturnStatement", "BinaryExpression", "LogicalExpression", "ConditionalExpression"}:
            return False
        if parent.type == "ExpressionStatement" and parent.expression == node:
            return True  # Explicit call with no usage of return value
    return False

def traverse_ast_7a(node, parent=None, js_code=None):
    """
    Recursively traverses the AST and finds all function calls with no return value.
    Returns a list of locations.
    """
    locations = []
    
    # Check if the current node is a function call with no return value
    if is_function_call_without_return(node, parent):
        pos = node.loc
        # Skip location at the very beginning
        if pos.start.line == 1 and pos.start.column == 0:
            return []
        
        # Extract the code snippet using the line and column locations
        if js_code:
            start_index = line_col_to_index(js_code, pos.start.line, pos.start.column)
            end_index = line_col_to_index(js_code, pos.end.line, pos.end.column)
            locations.append((start_index, end_index))
    
    # Recursively traverse child nodes
    for child_name, child in node.__dict__.items():
        if isinstance(child, list):  # List of child nodes
            for item in child:
                if isinstance(item, esprima.nodes.Node):
                    locations.extend(traverse_ast_7a(item, node, js_code))
        elif isinstance(child, esprima.nodes.Node):  # Single child node
            locations.extend(traverse_ast_7a(child, node, js_code))
    
    return locations

def replace_function_calls_with_no_return(js_code,ast_g):
    asm_code = []
    js_dict_list = []
    chunks_to_merge = []
    
    ast = ast_g
    # ast = esprima.parseScript(js_code,loc=True)
    
    def get_asm_code_f_call(start_ind):
        asm_code_f_call = f"""
        @external("js", "impFunc_{start_ind}")
        declare function impFunc_{start_ind}(): void;
    
        // Exported function in AssemblyScript
        export function f_{start_ind}(): void {{
            // Call the imported JavaScript function
            impFunc_{start_ind}();
        }}
        """
        return asm_code_f_call

    def get_js_dict_obj(called_function,start_ind):
        obj = f"""
        impFunc_{start_ind}: () => {{
            {called_function};
        }},
        """
        return obj

    def get_binding_code_f_call(start_ind):
        binding_code = f"""instance_lit.exports.f_{start_ind}();"""
        return binding_code
    
    # Get locations of function calls with no return value
    locations = traverse_ast_7a(ast,js_code=js_code)
    
    # Print results
    for loc in locations:
        called_function = js_code[loc[0]:loc[1]]
        asm_code.append(get_asm_code_f_call(loc[0]))
        js_dict_list.append(get_js_dict_obj(called_function,loc[0]))
        binding_code = get_binding_code_f_call(loc[0])
        chunks_to_merge.append({
        'code':binding_code,
        'start_ind': loc[0],
        'end_ind':loc[1]+1
        })

    return chunks_to_merge,asm_code,js_dict_list


# In[16]:


# rule 8 : class definitions

def find_class_nodes(node):
    if node is None:
        return []

    class_nodes = []
    if getattr(node, "type", None) in ("ClassDeclaration", "ClassExpression"):
        class_nodes.append(node)

    # Walk every attribute that might hold children
    for value in node.__dict__.values():
        if isinstance(value, list):
            for item in value:
                if hasattr(item, "type"):              # child node
                    class_nodes.extend(find_class_nodes(item))
        elif hasattr(value, "type"):                    # single child node
            class_nodes.extend(find_class_nodes(value))

    return class_nodes

def l_t_i(start_loc, end_loc, js_code):
    lines = js_code.split('\n')
    start_index = 0
    for i in range(start_loc.line - 1):
        start_index += len(lines[i]) + 1  # +1 for newline
    start_index += start_loc.column
    end_index = 0
    for i in range(end_loc.line - 1):
        end_index += len(lines[i]) + 1  # +1 for newline
    end_index += end_loc.column
    return start_index, end_index

def replace_class_defs(js_code,ast_g):
    ast = ast_g
    # ast = esprima.parseScript(js_code,loc=True)
    class_nodes = find_class_nodes(ast)

    chunks_to_merge,asm_code = [],[]

    def get_binding_code_class(class_name,class_type,start_index):
        binding_code = f'''
        const classContent_{start_index} = `
                ${{getString(instance_lit.exports.class_{start_index})}}
        `;
    
        const script_{start_index} = document.createElement("script");
        script_{start_index}.textContent = classContent_{start_index};
        document.body.appendChild(script_{start_index});
        '''
            
        return binding_code

    def get_asm_code_class(start_index,class_code):
        return f"""export let class_{start_index}: string = `{class_code}`;\n"""
    
    for item in class_nodes:
        class_name = item.id.name if hasattr(item, 'id') and item.id is not None else None
        class_type = item.type
        start_index, end_index = l_t_i(item.loc.start, item.loc.end, js_code)
        
        class_code = js_code[start_index:end_index]
        # print(class_code)
        asm_code_temp = get_asm_code_class(start_index,class_code)
        try:
            # compile_asm(asm_code_temp)
            asm_code.append(asm_code_temp)
            binding_code = get_binding_code_class(class_name,class_type,start_index)
            chunks_to_merge.append({
            'code':binding_code,
            'start_ind': start_index,
            'end_ind':end_index
            })
        except Exception as e:
            raise(e)
    
    return chunks_to_merge,asm_code,[]
    # return 0


# In[17]:


# rule 9 : function definitions
def find_function_definitions(ast_node, function_nodes=None):
    if function_nodes is None:
        function_nodes = []
    
    # Base case: Check if it's not an object or doesn't have a type attribute
    if not hasattr(ast_node, 'type'):
        return function_nodes
    
    # Check if current node is a function declaration or expression, but NOT a method definition
    if ast_node.type in ['FunctionDeclaration', 'FunctionExpression', 'ArrowFunctionExpression']:
        function_nodes.append(ast_node)
    
    # Recursively search all properties of the current node
    for attr_name in dir(ast_node):
        # Skip special attributes and methods
        if attr_name.startswith('__') or callable(getattr(ast_node, attr_name)):
            continue
        
        attr_value = getattr(ast_node, attr_name)
        
        # If the attribute is a list, check each item
        if isinstance(attr_value, list):
            for item in attr_value:
                find_function_definitions(item, function_nodes)
        # If the attribute might be an AST node, check it recursively
        elif hasattr(attr_value, 'type'):
            find_function_definitions(attr_value, function_nodes)
    
    return function_nodes

def l_t_i_f(start_loc, end_loc,js_code):
    lines = js_code.split('\n')
    start_index = 0
    for i in range(start_loc.line - 1):
        start_index += len(lines[i]) + 1  # +1 for newline
    start_index += start_loc.column
    end_index = 0
    for i in range(end_loc.line - 1):
        end_index += len(lines[i]) + 1  # +1 for newline
    end_index += end_loc.column
    return start_index, end_index
    
#######################################

def replace_func_defs(js_code,ast_g):
    ast = ast_g
    # ast = esprima.parseScript(js_code,loc=True)
    func_nodes = find_function_definitions(ast)

    chunks_to_merge,asm_code,js_dict_list = [],[],[]

    def get_binding_code_function(func_name,start_index,func_type):
        if func_type == 'FunctionDeclaration':
            binding_code = f"let {func_name} = instance_lit.exports.{func_name}"
        else:
            # ptint('pe')
            binding_code = f"instance_lit.exports.func_def_{start_index}"
        return binding_code
    
    for item in func_nodes:
        func_name = item.id.name if hasattr(item, 'id') and item.id is not None else None
        func_type = item.type
        # print(func_name)
        start_index, end_index = l_t_i_f(item.loc.start, item.loc.end,js_code)
        if func_type == 'FunctionDeclaration':
            user_msg = "Write the following JS function in AssemblyScript, and export it. Only provide the code, no explanation or use case:\n"
        else:
            user_msg = f"Write the following JS function in AssemblyScript, name it func_def_{start_index}, and export it. Only provide the code, no explanation or use case:\n"
        func_code = js_code[start_index:end_index]
        asm_code_temp = extract_assemblyscript_code(convert_js_to_asm_qwen(func_code,user_msg))
        try:
            compile_asm(asm_code_temp)
            asm_code.append(asm_code_temp)
            binding_code = get_binding_code_function(func_name,start_index,func_type)
            chunks_to_merge.append({
            'code':binding_code,
            'start_ind': start_index,
            'end_ind':end_index
            })
        except Exception as e:
            raise(e)
    
    return chunks_to_merge,asm_code,js_dict_list


# In[18]:


# rule 10: fp obf - memberExpression obfuscation

def replace_canvas_api_calls(js_code,ast_g):
    asm_code = []
    chunks_to_merge = []

    def _split_word(word: str) -> tuple:
        mid = len(word) // 2
        return word[:mid], word[mid:]

    def _get_asm_canvas(f_h, s_h, start_ind):
        return f'''
export const f_h_{start_ind}: string = "{f_h}";
export const s_h_{start_ind}: string = "{s_h}";
'''

    def node_to_dict(node):
        if hasattr(node, '__dict__'):
            return {key: node_to_dict(value) for key, value in node.__dict__.items()}
        elif isinstance(node, list):
            return [node_to_dict(item) for item in node]
        else:
            return node

    def _get_original_indices(substring, start_idx, end_idx, original_string):
        sliced_string = original_string[start_idx:end_idx]
        relative_idx = sliced_string.find(substring)
        if relative_idx == -1:
            return None
        original_start = start_idx + relative_idx
        original_end = original_start + len(substring)
        return original_start, original_end

    def _get_binding_code_canvas(f_h, s_h, start_ind, i_s):
        return f"[getString(instance_lit.exports.f_h_{start_ind})+getString(instance_lit.exports.s_h_{start_ind})]"

    def _canvas_l2i(js_code, line, column):
        lines = js_code.split('\n')
        return sum(len(lines[i]) + 1 for i in range(line - 1)) + column

    def _find_canvas_apis(js_code, node, positions):
        if getattr(node, 'type', None) == "MemberExpression":
            prop = node_to_dict(getattr(node, 'property', None))
            name = None
            access_type = None
            if not node.computed:
                access_type = '.'
                name = prop.get('name')
            elif prop.get('type') == 'Literal' and isinstance(prop.get('value'), str):
                access_type = '[]'
                name = prop.get('value')
                # print(name)

            if name and hasattr(node, 'loc'):
                start = node.loc.start
                end = node.loc.end
                s_index = _canvas_l2i(js_code, start.line, start.column)
                e_index = _canvas_l2i(js_code, end.line, end.column)
                positions.append({'start': s_index, 'end': e_index, 'target': name,'access_type' : access_type})

        for key in dir(node):
            if key.startswith('_'):
                continue
            value = getattr(node, key)
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, 'type'):
                        _find_canvas_apis(js_code, item, positions)
            elif hasattr(value, 'type'):
                _find_canvas_apis(js_code, value, positions)

    def _find_canvas_calls(js_code,ast_g):
        positions = []
        parsed = ast_g
        # parsed = esprima.parseScript(js_code, loc=True)
        _find_canvas_apis(js_code, parsed, positions)
        return positions

    calls = _find_canvas_calls(js_code,ast_g)

    for call in calls:
        indices = _get_original_indices(call['target'], call['start'], call['end'], js_code)
        if indices is None:
            continue

        start_ind, end_ind = indices
        f_h, s_h = _split_word(call['target'])

        i_s = call['access_type']

        asm_code.append(_get_asm_canvas(f_h, s_h, start_ind))
        binding_code = _get_binding_code_canvas(f_h, s_h, start_ind, i_s)

        if i_s == '.':
            start_ind -= 1
        elif i_s == '[]':
            start_ind=start_ind-2
            end_ind=end_ind+2

        chunks_to_merge.append({
            'code': binding_code,
            'start_ind': start_ind,
            'end_ind': end_ind
        })

    return chunks_to_merge, asm_code, []


# In[19]:


# rule 11: fp obf - dynamic codegen obfuscation

def replace_callee(js_code,ast_g):
    asm_code = []
    chunks_to_merge = []

    def node_to_dict(node):
        if hasattr(node, '__dict__'):
            return {key: node_to_dict(value) for key, value in node.__dict__.items()}
        elif isinstance(node, list):
            return [node_to_dict(item) for item in node]
        else:
            return node
    
    def _get_asm_cnvs_callee(cnvs_str,start_ind):
        return f"""
        export const e_call_{start_ind}:string = "eval";
        export const c_str_{start_ind}:string = {cnvs_str};
        """
    
    def _get_binding_code_callee(start_ind):
        return f"window[getString(instance_lit.exports.e_call_{start_ind})](getString(instance_lit.exports.c_str_{start_ind}))"
    
    def _canvas_l2i(js_code, line, column):
        lines = js_code.split('\n')
        return sum(len(lines[i]) + 1 for i in range(line - 1)) + column
    
    def _find_canvas_outer(js_code,ast_g):
        positions = []
        parsed = ast_g
        # parsed = esprima.parseScript(js_code, loc=True)
        _find_canvas_inner(js_code, parsed, positions)
        return positions
    
    def _find_canvas_inner(js_code, node, positions):
        # Match canvas() calls (CallExpression with callee.name == 'canvas')
        if getattr(node, 'type', None) == "MemberExpression":
            obj = node_to_dict(getattr(node, 'object', None))
            name = obj.get('name')
    
            target_properties = {'screen'}
    
            if name in target_properties and hasattr(node, 'loc'):
                start = node.loc.start
                end = node.loc.end
                s_index = _canvas_l2i(js_code, start.line, start.column)
                e_index = _canvas_l2i(js_code, end.line, end.column)
                positions.append({'start':s_index,'end':e_index, 'target':name})
        
        if getattr(node, 'type', None) == "CallExpression":
            callee = getattr(node, 'callee', None)
            # print(callee)
            if getattr(callee, 'type', None) == 'Identifier' and getattr(callee, 'name', '') == 'canvas':
            # if getattr(callee, 'name', '') == 'canvas':
                if hasattr(node, 'loc'):
                    start = node.loc.start
                    end = node.loc.end
                    s_index = _canvas_l2i(js_code, start.line, start.column)
                    e_index = _canvas_l2i(js_code, end.line, end.column)
                    positions.append({'start': s_index, 'end': e_index, 'target': 'canvas()'})
    
        # Recurse into children
        for key in dir(node):
            if key.startswith('_'):
                continue
            value = getattr(node, key)
            if isinstance(value, list):
                for item in value:
                    if hasattr(item, 'type'):
                        _find_canvas_inner(js_code, item, positions)
            elif hasattr(value, 'type'):
                _find_canvas_inner(js_code, value, positions)

    calls = _find_canvas_outer(js_code,ast_g)

    for call in calls:
        start_ind = call['start']
        end_ind = call['end']
        cnvs_str = js_code[start_ind:end_ind]

        asm_code.append(_get_asm_cnvs_callee(cnvs_str,start_ind))
        binding_code = _get_binding_code_callee(start_ind)

        chunks_to_merge.append({
            'code': binding_code,
            'start_ind': start_ind,
            'end_ind': end_ind
        })

    return chunks_to_merge, asm_code, []


# In[20]:


# rule 12: fp obf - regex based obfuscation

def replace_with_regex(js_code,ast_g):
    asm_code = [
        '''
        export const cv1_poka: string = "can";
        export const cv2_poka: string = "vas";
        '''
    ]
    binding_code = "getString(instance_lit.exports.cv1_poka)+getString(instance_lit.exports.cv2_poka)"

    pattern = r"""((['"]|\\)+canvas(['"]|\\)+)"""

    chunks_to_merge = []

    for m in re.finditer(pattern, js_code):
        matched_text = m.group(0)
        stripped = re.sub(r"^['\"\\]+|['\"\\]+$", "", matched_text)
        if stripped == "canvas":
            post = m.end()
            while js_code[post].isspace():
                post+=1
            if js_code[post]==':':
                continue
            chunks_to_merge.append({
                'code': binding_code,
                'start_ind': m.start(),
                'end_ind': m.end()
            })

    return chunks_to_merge, asm_code, []
    
    


# In[21]:


# rule 13: fp obf - screen memberexp with obf prop name

def replace_obf_screen(js_code,ast_g):
    
    def node_to_dict(node):
        if hasattr(node, '__dict__'):
            return {key: node_to_dict(value) for key, value in node.__dict__.items()}
        elif isinstance(node, list):
            return [node_to_dict(item) for item in node]
        else:
            return node
    
    def canvas_l2i(js_code, line, column):
        lines = js_code.split('\n')
        return sum(len(lines[i]) + 1 for i in range(line - 1)) + column
    
    def rule_check_poi(js_code, node, positions):
        if getattr(node, 'type', None) == "MemberExpression":
            obj = node_to_dict(getattr(node, 'object', None))
            name = obj.get('name')
    
            target_properties = {'screen','canvas'}
    
            if name in target_properties and hasattr(node, 'loc'):
                start = node.loc.start
                end = node.loc.end
                s_index = canvas_l2i(js_code, start.line, start.column)
                e_index = canvas_l2i(js_code, end.line, end.column)
                positions.append({'start':s_index,'end':e_index, 'target':name})
    
        # Recurse into children
        for key in dir(node):
            if key.startswith('_'):
                continue  # Skip internal attributes
            value = getattr(node, key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) or hasattr(item, 'type'):
                        rule_check_poi(js_code, item, positions)
            elif isinstance(value, dict) or hasattr(value, 'type'):
                rule_check_poi(js_code, value, positions)
    
    positions = []
    parsed = ast_g
    # parsed = esprima.parseScript(js_code, loc=True)
    
    rule_check_poi(js_code, parsed, positions)
    
    chunks_to_merge = []
    
    asm_code = [
        f"""
        export const sc1_poi: string = "scr";
        export const sc2_poi: string = "een";
        """
    ]
    
    binding_code = f"window[getString(instance_lit.exports.sc1_poi)+getString(instance_lit.exports.sc2_poi)]"
    
    for i in positions:
        start_ind = i['start']
        end_ind = i['start']+6
        chunks_to_merge.append({
                'code': binding_code,
                'start_ind': start_ind,
                'end_ind': end_ind
        })

    return chunks_to_merge, asm_code, []


# In[22]:

async def validate_js_code(code):
    """Validate JavaScript code using syntax check and headless browser test"""
    # Step 1: Check syntax
    syntax_valid = check_js_syntax(code)
    if not syntax_valid:
        return {"valid": False, "errors": ["Syntax error in converted code"]}
    
    # Step 2: Test in browser
    browser_test_result = await test_in_browser(code)
    # browser_test_result = test_in_browser(code)
    return browser_test_result

def check_js_syntax(code):
    """Check JavaScript syntax using esprima"""
    try:
        esprima.parseScript(code)
        return True
    except Exception:
        return False

async def test_in_browser(code):
    """Test JavaScript code in a headless browser asynchronously"""
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            # Set a default timeout if needed
            page.set_default_timeout(5000)
            
            # Create a clean test page
            await page.set_content("<html><body><script></script></body></html>")
            
            # Setup error detection
            errors = []
            page.on("pageerror", lambda err: errors.append(str(err)))
            page.on("console", lambda msg: errors.append(str(msg)) if msg.type == "error" else None)
            
            # Inject and execute the code
            try:
                # Wrap code execution in a try-catch to detect runtime errors
                wrapped_code = f"""
                try {{
                    {code}
                    window.__TEST_RESULT__ = true;
                }} catch (error) {{
                    window.__TEST_RESULT__ = false;
                    window.__TEST_ERROR__ = error.message;
                    console.error(error);
                }}
                """
                # No timeout parameter here
                await page.evaluate(wrapped_code)
                
                # Check if execution was successful
                result = await page.evaluate("window.__TEST_RESULT__")
                if not result:
                    error_msg = await page.evaluate("window.__TEST_ERROR__")
                    errors.append(error_msg)
            except Exception as e:
                errors.append(str(e))
            
            await browser.close()
            return {"valid": len(errors) == 0, "errors": errors}
        except Exception as e:
            return {"valid": False, "errors": [f"Browser test setup failed: {str(e)}"]}


# In[23]:


async def single_rule_check(js_code,rule_fn, ast_g):
    js_chunks,asm_chunks,js_dict_list = rule_fn(js_code,ast_g)
    mod_js,cnv_ptg = get_mod_js(js_code,js_chunks)
    asm_code = "".join(asm_chunks)
    if js_dict_list:
        js_dict = "{"+"".join(js_dict_list)+"}"
    else:
        js_dict=None

    converted_code = get_wasm_loading_code(asm_code,mod_js, has_string=False,js_dict=js_dict)
    validation_result = await validate_js_code(converted_code)
    # if validation_result['valid']==False:
    #     print(converted_code)
    return validation_result

def wasm_obfuscation(js_code,rules,ast_g):
    js_chunks_temp,asm_chunks_temp,js_dict_list_temp,js_chunks,asm_chunks,js_dict_list = [],[],[],[],[],[]
    for rule in rules:
        try:
            js_chunks_temp,asm_chunks_temp,js_dict_list_temp = rule(js_code,ast_g)
        except:
            print('why is this happening!!!!!!!!!!!!!!!')
        js_chunks.extend(js_chunks_temp)
        asm_chunks.extend(asm_chunks_temp)
        js_dict_list.extend(js_dict_list_temp)
    mod_js,cnv_ptg = get_mod_js(js_code,js_chunks)
    asm_code = "".join(asm_chunks)
    if js_dict_list:
        js_dict = "{"+"".join(js_dict_list)+"}"
    else:
        js_dict=None

    return get_wasm_loading_code(asm_code,mod_js, has_string=False,js_dict=js_dict),cnv_ptg



# In[25]:


#code to fix freeze
import gc
import functools
import sys
import textwrap
from asyncio.subprocess import PIPE

# process_pool = concurrent.futures.ProcessPoolExecutor(mp.cpu_count())
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# async def obfuscate_with_timeout(code, rules, timeout, ast_g):
#     loop = asyncio.get_running_loop()
#     fut = loop.run_in_executor(process_pool, wasm_obfuscation, code, rules, ast_g)
#     try:
#         return await asyncio.wait_for(fut, timeout)
#     except asyncio.TimeoutError:
#         fut.cancel()            # mark future cancelled
#         raise 

_CHILD_SCRIPT = textwrap.dedent("""
    import sys, esprima
    code = sys.stdin.read()
    try:
        esprima.parseScript(code, loc=True)
        sys.exit(0)          # success
    except Exception:
        sys.exit(1)          # syntax/tokenizer error
""")

async def can_parse_in_time(code: str, timeout: float) -> bool:
    """
    Return True iff `esprima.parseScript(code, loc=True)` finishes without
    exception *and* within `timeout` seconds.  Otherwise return False.

    A subprocess is hard killed if it overruns the timeout.
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", _CHILD_SCRIPT,
        stdin=PIPE, stdout=PIPE, stderr=PIPE,
    )

    try:
        # just wait for the child to exit (it prints nothing)
        await asyncio.wait_for(proc.communicate(code.encode()), timeout)
    except asyncio.TimeoutError:
        proc.kill()          # guaranteed termination
        await proc.wait()
        return False         # took too long

    return proc.returncode == 0


# In[ ]:
    

async def main():
    # print("started code")
    # rules = [replace_literals_recursive,obfuscate_functions,replace_int_arrays,replace_float_arrays,replace_class_defs,replace_func_defs,
    #         replace_if_else,replace_for_loops,replace_while_loops,replace_function_calls_with_no_return,
    #             replace_canvas_api_calls, replace_callee, replace_with_regex,replace_obf_screen]
    # mahsin
    rules = [replace_literals_recursive]
    rule_names = [rule.__name__ for rule in rules]

    csv_fields = [
        'categories', 'cnv_ptg', 'time_cnv', 'time_val',
        'working_rules', 'abrupt_fail','time_fail','ast_fail', 'large_file_skip','validation_fail', 'success'
    ]

    # buffer = []
    buffer_size = 10  # batch size

    def flush_buffer(buffer, csv_file, write_header=False):
        if not buffer:
            return
        mode = 'a'
        file_already_exists = os.path.isfile(csv_file) and os.path.getsize(csv_file) > 0
        with open(csv_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            if write_header and not file_already_exists:
                writer.writeheader()
            writer.writerows(buffer)
        buffer.clear()
    # for handling csv ###############################

    samples_to_convert = glob.glob('/samples_for_conversion/*.json')
    
    # print(samples_to_convert)
    #[8:9]
    # for sample in tqdm(samples_to_convert[0]):
    for sample in tqdm(['/samples_for_conversion/sample_02_20250528_133938.json']):
        # base_name = os.path.basename(sample).split('.')[0]
        base_name = 'replace_literals_recursive'
        js={}
        wasm={}
        with open(sample,'r') as f:
            data = json.load(f)

        buffer = []
        # csv_file = f''
        csv_file = f'/persistent_data/conversion/conversion_logs/ablation_study/conversion_log_{base_name}.csv'
        write_header=True
        
        unparseable_count=0
        json_count = 0

        start_index = 0
        count = start_index
        # print("before starting loop")
        for i in tqdm(range(start_index, len(data)), initial=start_index, total=len(data)):
            count += 1

            cnv_ptg = None
            time_cnv = None
            time_val = None
            working_rules = []
            categories=[]
            abrupt_fail = 0
            time_fail = 0
            ast_fail=0
            large_file_skip=0
            validation_fail = 0
            success = 0
            # if sys.getsizeof(data[i]['script'])>1128658:
            if sys.getsizeof(data[i]['script'])>800000:
                large_file_skip=1
            if count==200 or count==431:
                large_file_skip=1

            timeout_secs = 600  # 10 minutes
            deadline = time.time() + timeout_secs
            
            if large_file_skip==0:
                item=data[i]
                # print("selected item")
                code = item['script']
                try:
                    categories = item['categories']
                except:
                    categories = []
                # print("category found")
                # print("before parse")

                good_ast = await can_parse_in_time(code,10)
                if good_ast:
                    ast_g = esprima.parseScript(code,loc=True)
                else:
                    ast_fail=1
                    unparseable_count += 1
                    print(f'{unparseable_count} unparseable code(s) so far')
            # print("after parse")
            # rule check
            if ast_fail==0 and large_file_skip==0:
                for rule in rules:
                    try:
                        remaining_time = deadline - time.time()
                        if remaining_time <= 0:
                            raise asyncio.TimeoutError()
                        # print("before single rule check")
                        v = await single_rule_check(code, rule, ast_g)
                        # print("after single rule check")
                        if v['valid']:
                            working_rules.append(rule)
                    except asyncio.TimeoutError:
                        time_fail=1
                        break
                    except:
                        pass
        
            # conversion and validation
            if time.time() > deadline:
                time_fail=1
        
            if working_rules and time_fail==0:
                try:
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        raise asyncio.TimeoutError()
                    start_time = time.time()
                    # converted_code, cnv_ptg = await obfuscate_with_timeout(
                    #     code, working_rules, remaining_time, ast_g
                    # )
                    # print("before conversion")
                    converted_code, cnv_ptg = wasm_obfuscation(code,working_rules,ast_g)
                    time_cnv = round(time.time() - start_time, 4)
                    # print("after conversion")
        
                    remaining_time = deadline - time.time()
                    if remaining_time <= 0:
                        raise asyncio.TimeoutError()
                    start_time = time.time()
                    # print("before validation")
                    v = await validate_js_code(converted_code)
                    # print("after validation")
                    # v = await asyncio.wait_for(validate_js_code(converted_code), timeout=remaining_time)
                    time_val = round(time.time() - start_time, 4)
        
                    label = 1 if categories else 0
                    if v['valid']:
                        js[count] = {'label': label, 'content': code}
                        wasm[count] = {'label': label, 'content': converted_code}
                        success = 1
                    else:
                        validation_fail = 1
                except asyncio.TimeoutError:
                    time_fail=1
                    # break
                except Exception as e:
                    
                    abrupt_fail = 1
        
            # logging
            buffer.append({
                'categories': "|".join(categories),
                'cnv_ptg': cnv_ptg,
                'time_cnv': time_cnv,
                'time_val': time_val,
                'working_rules': "|".join([r.__name__ for r in working_rules]),
                'abrupt_fail': abrupt_fail,
                'time_fail':time_fail,
                'ast_fail':ast_fail,
                'large_file_skip':large_file_skip,
                'validation_fail': validation_fail,
                'success': success
            })
            if len(buffer) >= buffer_size:
                flush_buffer(buffer, csv_file, write_header)
                write_header = False
            if count % 50 == 0:
                print(f'Processed {count} scripts')
                json_count=count//50
                # with open(f'/conversion_output/js_{base_name}_{json_count}.json','w') as f:
                with open(f'/conversion_output/ablation_study/js_{base_name}_{json_count}.json','w') as f:
                    json.dump(js,f)
                # with open(f'/conversion_output/wasm_{base_name}_{json_count}.json','w') as f:
                with open(f'/conversion_output/ablation_study/wasm_{base_name}_{json_count}.json','w') as f:
                    json.dump(wasm,f)
                js={}
                wasm={}
                # flush_buffer(buffer, csv_file, write_header)
                gc.collect()



if __name__ == "__main__":
    asyncio.run(main())