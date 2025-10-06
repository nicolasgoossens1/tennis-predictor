from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import csv
import time
import os

# Define the output folder for CSV files (relative to your project)
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
os.makedirs(DATA_FOLDER, exist_ok=True)  # Ensure the folder exists

# Function to scrape a single URL
def scrape_tennis_data(url, output_filename):
    # 1. Set up Selenium WebDriver with options
    from selenium.webdriver.chrome.options import Options
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in background
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)

        # 2. Wait for the table to load
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.ID, "reportable")))

        # 3. Parse the Rendered HTML with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # 4. Locate the Outer Table (maintable)
        maintable = soup.find('table', {'id': 'reportable'})
        if not maintable:
            raise Exception("Table with id 'reportable' not found on the page.")

        # 6. Extract Headers Dynamically
        headers = []
        thead = maintable.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                for th in header_row.find_all('th'):
                    headers.append(th.get_text(strip=True))
        
        if not headers:
            raise Exception("Header row not found in the table.")

        print(f"Extracted Headers for {url}: {headers}")

        # 7. Extract Table Rows
        rows_data = []
        tbody = maintable.find('tbody')
        if tbody:
            for row in tbody.find_all('tr'):
                cells = row.find_all('td')
                if cells:
                    row_values = [cell.get_text(strip=True) for cell in cells]
                    if row_values:  # Only add non-empty rows
                        rows_data.append(row_values)

        # 8. Write to CSV in the 'data' folder
        csv_filename = os.path.join(DATA_FOLDER, output_filename)
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows_data)

        print(f"Data successfully written to {csv_filename}")
        print(f"Scraped {len(rows_data)} rows of data")

    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
    finally:
        # 9. Close the WebDriver
        if driver is not None:
            driver.quit()

# List of URLs to scrape
urls = [
    "https://tennisabstract.com/reports/mcp_leaders_serve_men_last52.html",
    "https://tennisabstract.com/reports/mcp_leaders_return_men_last52.html",
    "https://tennisabstract.com/reports/mcp_leaders_rally_men_last52.html",
    "https://tennisabstract.com/reports/mcp_leaders_tactics_men_last52.html"
]

# Scrape each URL and save to a separate CSV file
if __name__ == "__main__":
    filenames = ["serve_leaders.csv", "return_leaders.csv", "rally_leaders.csv", "tactics_leaders.csv"]
    
    for url, filename in zip(urls, filenames):
        print(f"Scraping {url}...")
        scrape_tennis_data(url, filename)
        time.sleep(2)  # Be respectful to the server