from playwright.sync_api import sync_playwright
import json

FP_APIS = [
    "HTMLCanvasElement.prototype.toDataURL",
    "HTMLCanvasElement.prototype.getContext",
    "CanvasRenderingContext2D.prototype.getImageData",
    "AudioContext.prototype.createAnalyser",
    "navigator.mediaDevices.getUserMedia",
    "RTCPeerConnection.prototype.createDataChannel",
    "RTCPeerConnection.prototype.createOffer",
    "Navigator.prototype.userAgent",
    "Navigator.prototype.language"
]

# def inject_monitoring_script(apis):
#     hooks = []
#     for api in apis:
#         hooks.append(f"""
#         (()=>{{
#           try {{
#             const path = "{api}".split('.');
#             let obj = globalThis;
#             for (let i=0;i<path.length-1;i++) obj = obj[path[i]];
#             const key = path[path.length-1];
#             const orig = obj[key];

#             if (typeof orig === 'function') {{
#               obj[key] = function(...args){{
#                 globalThis.__fp_logs.push("{api}");
#                 return orig.apply(this,args);
#               }};
#             }} else {{
#               Object.defineProperty(obj,key,{{
#                 get(){{
#                   globalThis.__fp_logs.push("{api}");
#                   return orig;
#                 }},
#                 configurable:true
#               }});
#             }}
#           }} catch(e){{ /* ignore missing APIs */ }}
#         }})();
#         """)
#     return f"globalThis.__fp_logs = [];\n{''.join(hooks)}"

def inject_monitoring_script(apis):
    fn_hooks    = []   # ordinary functions
    nav_props   = []   # navigator.xxx
    screen_props = []  # screen.xxx

    for api in apis:
        if api.startswith("Navigator.prototype."):
            nav_props.append(api.split('.')[-1])
        elif api.startswith("Screen.prototype."):
            screen_props.append(api.split('.')[-1])
        else:  # functions we can monkey-patch safely
            fn_hooks.append(api)

    # 1 function & method hooks (Canvas, AudioContext …)
    fn_js = ""
    for api in fn_hooks:
        fn_js += f"""
        (()=>{{
          const path = "{api}".split('.');
          let obj = globalThis;
          for (let i=0;i<path.length-1;i++) obj = obj[path[i]];
          const key = path[path.length-1];
          const orig = obj[key];
          if (typeof orig === 'function') {{
             obj[key] = function(...args){{
               globalThis.__fp_logs.push("{api}");
               return orig.apply(this,args);
             }};
          }}
        }})();
        """

    # 2 navigator property hooks (userAgent, language …)
    nav_js = ""
    if nav_props:
        props = ','.join(f'"{p}"' for p in nav_props)
        nav_js = f"""
        (()=>{{
          const props = new Set([{props}]);
          props.forEach(prop => {{
            const origGetter = Navigator.prototype.__lookupGetter__(prop);
            if (!origGetter) return;  // safety
            Object.defineProperty(globalThis.navigator, prop, {{
              get(){{
                globalThis.__fp_logs.push("Navigator.prototype."+prop);
                return origGetter.call(globalThis.navigator);
              }},
              configurable:true   // so pages can still redefine if they want
            }});
          }});
        }})();
        """

    # 3 screen property hooks (width, height …) – same idea
    screen_js = ""
    if screen_props:
        props = ','.join(f'"{p}"' for p in screen_props)
        screen_js = f"""
        (()=>{{
          const props = new Set([{props}]);
          props.forEach(prop => {{
            const desc = Object.getOwnPropertyDescriptor(Screen.prototype, prop);
            if (!desc || typeof desc.get !== 'function') return;
            Object.defineProperty(globalThis.screen, prop, {{
              get(){{
                globalThis.__fp_logs.push("Screen.prototype."+prop);
                return desc.get.call(globalThis.screen);
              }},
              configurable:true
            }});
          }});
        }})();
        """

    return (
        "globalThis.__fp_logs = [];\n"
        + fn_js
        + nav_js
        + screen_js
    )

def monitor_js(js_code):
    with sync_playwright() as p:
        browser  = p.chromium.launch(headless=True)
        context  = browser.new_context()
        # inject into **every** new document (about:blank included)
        context.add_init_script(inject_monitoring_script(FP_APIS))

        page = context.new_page()           # still on about:blank – DOM already exists
        page.evaluate(js_code)              # run the user-supplied JS
        logs = page.evaluate("globalThis.__fp_logs")
        browser.close()
        return logs

if __name__ == "__main__":
    test_js = """
    const c = document.createElement('canvas');
    const ctx = c.getContext('2d');
    ctx.fillText('hi',10,10);
    c.toDataURL();

    const ac = new AudioContext();
    ac.createAnalyser();

    navigator.userAgent;
    navigator.language;
    """
    print("Accessed fingerprinting APIs:")
    print(json.dumps(monitor_js(test_js), indent=2))
