import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    # We use a persistent context to save your login session. 
    # This means you only have to log in manually the first time.
    user_data_dir = os.path.join(os.getcwd(), "ashoka_browser_data")
    
    print("Starting Playwright...")
    async with async_playwright() as p:
        # headless=False is required for the first run so you can interact with the Google Login.
        # Once you're logged in, you can try changing headless=True if you prefer.
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False, 
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        page = await browser.new_page()
        url = "https://sg.ashoka.edu.in/platform/sports/aba"
        
        print(f"Navigating to {url}...")
        await page.goto(url)

        # Check if we are redirected to a login page
        if "login" in page.url or "accounts.google.com" in page.url:
            print("Login required.")
            
            # Attempt to auto-click the 'Continue with Google' button if on the platform login page
            if "login" in page.url:
                try:
                    print("Attempting to click 'Continue with Google'...")
                    # We look for a button or text that matches the common Google login prompt
                    # Increase timeout slightly if network is slow
                    await page.get_by_text("Continue with Google", exact=False).click(timeout=5000)
                except Exception as e:
                    print("Could not auto-click the button. Please click it manually.")
                    
            print("Please finish logging in with your @ashoka.edu.in account in the opened browser window.")
            print("Waiting for you to complete the login...")
            
            # Wait until the URL changes to the main sports platform URL
            # timeout=0 means it will wait indefinitely until you finish logging in
            await page.wait_for_url("**/platform/sports/aba**", timeout=0)
            
        print("Successfully loaded the target page!")
        
        # Wait a short moment for the React application to render the data
        # Since 'networkidle' can hang indefinitely on modern SPAs that constantly poll APIs, we wait 3 seconds instead
        await page.wait_for_timeout(3000)
        
        # Here you can add your specific scraping logic. 
        # Since the page is hidden behind a login, you'll need to inspect the page structure
        # once logged in to write the exact selectors.
        
        # Example: Get the page title
        title = await page.title()
        print(f"Page Title: {title}")
        
        # Example: Scrape all text from the main body
        body_text = await page.locator("body").inner_text()
        print("Page Text snippet (first 500 chars):")
        print(body_text[:500])
        
        # Example: Extract table rows if there are tables
        # rows = await page.locator("tr").all_inner_texts()
        # for row in rows:
        #     print(row)

        print("Finished scraping. Closing browser...")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
