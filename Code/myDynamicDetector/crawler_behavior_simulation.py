import asyncio
import json
import argparse
import random
import math
from playwright.async_api import async_playwright

async def random_mouse_movements(page, num_moves=20):
    """
    Simulate random mouse movements within the visible viewport.
    """
    viewport = page.viewport_size or {"width": 1280, "height": 720}
    width, height = viewport["width"], viewport["height"]

    print("Simulating random mouse movements...")
    for _ in range(num_moves):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        await page.mouse.move(x, y, steps=random.randint(5, 25))
        await asyncio.sleep(random.uniform(0.1, 0.5))


async def auto_scroll(page, scroll_pause=0.5, max_scrolls=10):
    """
    Scroll down the page gradually with random distances to trigger lazy-loaded or delayed scripts.
    """
    print("Scrolling page with random distances...")
    for _ in range(max_scrolls):
        # Randomize scroll distance between half and full viewport height
        await page.evaluate("""
            const scrollDistance = Math.floor(
                Math.random() * (window.innerHeight - window.innerHeight / 2) + window.innerHeight / 2
            );
            window.scrollBy(0, scrollDistance);
        """)
        await asyncio.sleep(random.uniform(scroll_pause, scroll_pause + 0.5))

    # Optionally scroll back to top at the end
    await page.evaluate("window.scrollTo(0, 0);")
    print("Randomized scrolling complete.")


async def main(target_url: str, output_file: str):
    """
    Main function to run the Playwright crawler.
    """
    # target_url = "https://sprite.utsa.edu/WASM_Fp/Environement_Sysinfo/navigator-screen-fingerprint/navigator-screen-fingerprint.html"
    # output_file = "javascript_with_hash_partition_1.json"
    instrument_script_path = "./instrument_dynamic_prefix.js"

    async with async_playwright() as p:
        # 1. Launch the browser and create a new page with a realistic user-agent
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36'
        )
        page = await context.new_page()

        try:
            await page.add_init_script(path=instrument_script_path)
            print(f"Injecting script from: {instrument_script_path}")

            # 2. Navigate to the target URL
            print(f"Navigating to: {target_url}")
            await page.goto(target_url, wait_until="networkidle")

            # 3. Wait for a few seconds to allow async scripts to execute
            # await page.wait_for_timeout(3000)
            await page.wait_for_selector('body')

            # Simulate user interactions
            await random_mouse_movements(page, num_moves=30)
            await auto_scroll(page, scroll_pause=0.5, max_scrolls=15)

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
                    # "crawl_id": 1,
                    # "visit_id": 1,
                    # "script_url": target_url
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
