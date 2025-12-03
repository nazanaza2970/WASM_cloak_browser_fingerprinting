import asyncio
import json
import argparse
from playwright.async_api import async_playwright

async def main(target_url: str, output_file: str):
    """
    Main function to run the Playwright crawler.
    """
    # target_url = "https://sprite.utsa.edu/WASM_Fp/Environement_Sysinfo/navigator-screen-fingerprint/navigator-screen-fingerprint.html"
    # output_file = "javascript_with_hash_partition_1.json"
    instrument_script_path = "./instrument_dynamic_prefix.js"

    async with async_playwright() as p:
        # Launch a browser (Chromium is a good default)
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            # 1. Inject the instrumentation script before any page navigation.
            # This ensures it runs before any website scripts.
            await page.add_init_script(path=instrument_script_path)
            print(f"Injecting script from: {instrument_script_path}")

            # 2. Navigate to the target URL
            print(f"Navigating to: {target_url}")
            await page.goto(target_url, wait_until="networkidle")

            # 3. Wait for a few seconds to allow async scripts to execute
            await page.wait_for_timeout(3000)

            # 4. Extract the logs from the page's global scope
            api_logs = await page.evaluate("() => window.__API_LOGS__")
            print(f"Collected {len(api_logs)} API calls.")

            # 5. Format the data into the structure your Python script expects.
            # NOTE: In a real system, you would intercept network requests
            # to get individual script URLs and their content hashes.
            # Here, we use a placeholder for simplicity.
            formatted_data = {
                "placeholder_content_hash": {
                    "information": api_logs if api_logs else [],
                    "content_hash": "placeholder_content_hash",
                    "crawl_id": 1,
                    "visit_id": 1,
                    "script_url": target_url
                }
            }

            # 6. Save the data to a JSON file
            with open(output_file, 'w') as f:
                json.dump(formatted_data, f, indent=4)
            print(f"Data successfully saved to: {output_file}")

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # Ensure the browser is always closed
            await browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playwright Crawler with Instrumentation")
    parser.add_argument('--url', type=str, default="https://sprite.utsa.edu/WASM_Fp/Graphics_rendering/canvas-fingerprint/canvas-fingerprint.html", help='Target URL to crawl')
    parser.add_argument('--output', type=str, default="javascript_with_hash_partition_1.json", help='Output JSON file')
    args = parser.parse_args()
    asyncio.run(main(args.url, args.output))
