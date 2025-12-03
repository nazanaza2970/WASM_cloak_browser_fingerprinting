import random
import time
from playwright.sync_api import sync_playwright, Page

def simulate_user_actions(page: Page):
    """Simulates random mouse movements and scrolling on a page."""
    
    print("🤖 Simulating user behavior...")

    # Get the size of the viewport to keep mouse movements within bounds
    viewport_size = page.viewport_size
    if not viewport_size:
        print("Could not get viewport size.")
        return
        
    width, height = viewport_size['width'], viewport_size['height']

    # --- Simulate Random Mouse Movements ---
    print("   - Moving mouse randomly...")
    for _ in range(15):  # Perform 15 random mouse moves
        random_x = random.randint(0, width - 1)
        random_y = random.randint(0, height - 1)
        # The 'steps' argument makes the movement appear smoother
        page.mouse.move(random_x, random_y, steps=5)
        time.sleep(random.uniform(0.1, 0.4))

    # --- Simulate Scrolling ---
    print("   - Scrolling down the page...")
    for _ in range(5):  # Scroll down 5 times
        scroll_amount = random.randint(400, 800)
        # Use the mouse wheel to scroll down
        page.mouse.wheel(0, scroll_amount)
        time.sleep(random.uniform(0.8, 1.5))

def main():
    """Main function to run the Playwright automation."""
    with sync_playwright() as p:
        # Launch the browser. Set headless=False to watch the script in action.
        browser = p.chromium.launch(headless=False, slow_mo=50)
        context = browser.new_context()

        # --- Start Tracing ---
        # This is the core part for capturing the trace.
        # We enable screenshots, snapshots, and source code recording.
        print("🔴 Starting trace recording...")
        context.tracing.start(screenshots=True, snapshots=True, sources=True)

        page = context.new_page()
        page.goto("http://localhost:8000/one_js_one_wasm.html")  # Replace with your target URL

        # Run the simulation function
        simulate_user_actions(page)
        
        # --- Stop Tracing ---
        # The trace is saved to the specified path upon stopping.
        trace_path = "trace.zip"
        print(f"⏹️ Stopping trace recording. Saving to '{trace_path}'...")
        context.tracing.stop(path=trace_path)

        print(f"✅ Trace saved successfully! You can view it with 'playwright show-trace {trace_path}'")
        browser.close()

if __name__ == "__main__":
    main()