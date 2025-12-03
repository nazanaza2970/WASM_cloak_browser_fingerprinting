import re
# import argparse

def html_generator_conversion(code_string: str) -> str:
    """
    Generate an HTML sandbox that can execute the provided JavaScript code string,
    automatically creating missing DOM elements (with canvas-aware proxies) as needed.
    """
    def find_element_ids(js_code: str) -> set:
        """
        Finds all element IDs referenced by getElementById or querySelector('#...').
        """
        # Regex for document.getElementById('some-id')
        by_id = re.findall(r"document\.getElementById\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", js_code)
        
        # Regex for document.querySelector('#some-id')
        query_id = re.findall(r"document\.querySelector(?:All)?\s*\(\s*['\"]#([^'\" ]+)['\"]\s*\)", js_code)
        
        return set(by_id + query_id)

    def find_element_classes(js_code: str) -> set:
        """
        Finds all element class names referenced by getElementsByClassName or querySelector('. ...').
        """
        # Regex for document.getElementsByClassName('some-class')
        by_class = re.findall(r"document\.getElementsByClassName\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", js_code)
        
        # Regex for document.querySelector('.some-class')
        # This handles simple cases and avoids capturing more complex selectors
        query_class = re.findall(r"document\.querySelector(?:All)?\s*\(\s*['\"]\s*\.([^'\" ]+)['\"]\s*\)", js_code)
        
        # Combine lists and split multi-class strings (e.g., "btn btn-primary")
        all_classes = set()
        for item in by_class + query_class:
            all_classes.update(item.split())
            
        return all_classes

    def find_element_tags(js_code: str) -> set:
        """
        Finds all element tags referenced by getElementsByTagName.
        """
        # Regex for document.getElementsByTagName('div')
        by_tag = re.findall(r"document\.getElementsByTagName\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", js_code)
        return set(by_tag)

    def create_placeholder_elements(ids: set, classes: set, tags: set) -> str:
        """
        Generates HTML placeholder elements from sets of IDs, classes, and tags.
        """
        elements = []
        
        # 1. Create elements for IDs (most specific)
        for element_id in ids:
            if 'canvas' in element_id.lower():
                elements.append(f'<canvas id="{element_id}"></canvas>')
            elif 'svg' in element_id.lower():
                elements.append(f'<svg id="{element_id}"></svg>')
            else:
                elements.append(f'<div id="{element_id}"></div>')
                
        # 2. Create generic <div> elements for each class
        for element_class in classes:
            elements.append(f'<div class="{element_class}"></div>')
            
        # 3. Create elements for each tag
        # Avoid creating tags that are already in the template or were found as IDs/classes
        safe_tags = tags - ids - classes - {'html', 'head', 'body', 'script', 'title', 'h1'}
        for element_tag in safe_tags:
            elements.append(f'<{element_tag}></{element_tag}>')
            
        if not elements:
            return ""
            
        return "\n    ".join(elements)

    # 1. Find all DOM dependencies
    required_ids = find_element_ids(code_string)
    required_classes = find_element_classes(code_string)
    required_tags = find_element_tags(code_string)

    print(f"Found IDs: {required_ids}")
    print(f"Found Classes: {required_classes}")
    print(f"Found Tags: {required_tags}")

    # 2. Create placeholder HTML for the discovered elements
    placeholder_elements = create_placeholder_elements(required_ids, required_classes, required_tags)

    html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dynamic Analysis Sandbox</title>
        </head>
        <body>
            <h1>Executing script...</h1>
            
            {placeholder_elements}
            
            <script>
            // To prevent errors if the script runs before the DOM is fully ready
            document.addEventListener('DOMContentLoaded', (event) => {{
                try {{
                    {code_string}
                }} catch (e) {{
                    console.error('Script execution failed:', e);
                }}
            }});
            </script>
        </body>
        </html>
        """

    return html_content
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Generate HTML Sandbox for JS Code")
#     parser.add_argument('--code', type=str, required=True, help='JavaScript code string to embed in the sandbox')



