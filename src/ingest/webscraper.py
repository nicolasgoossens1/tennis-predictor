from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import csv
import time
import os

# Define the output folder for CSV files
DATA_FOLDER = r"C:\Users\nicol\Downloads\ai_proj\tennis-predictor\data"
os.makedirs(DATA_FOLDER, exist_ok=True)  # Ensure the folder exists

# Function to scrape a single URL
def scrape_tennis_data(url, output_filename):
    # 1. Set up Selenium WebDriver
    driver = webdriver.Chrome()  # Ensure you have ChromeDriver installed and in PATH
    driver.get(url)

    # 2. Wait for the Page to Load
    time.sleep(3)  # Adjust if necessary

    # 3. Parse the Rendered HTML with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # 4. Locate the Outer Table (maintable)
    maintable = soup.find('table', {'id': 'reportable'})
    if not maintable:
        driver.quit()
        raise Exception("Table with id 'reportable' not found on the page.")

    # # 5. Locate the Inner Table (matches) Inside maintable
    # matches_table = maintable.find('table', {'id': 'matches'})
    # if not matches_table:
    #     driver.quit()
    #     raise Exception("Table with id 'matches' not found inside 'maintable'.")

    # 6. Extract Headers Dynamically
    headers = []
    header_row = maintable.find('thead').find('tr')  # Locate the header row
    if header_row:
        for th in header_row.find_all('th'):
                            headers.append(th.get_text(strip=True))  # Directly extract the text inside the <th>
    else:
        driver.quit()
        raise Exception("Header row not found in the table.")

    print(f"Extracted Headers for {url}: {headers}")

    # 7. Extract Table Rows
    rows_data = []
    for row in maintable.find('tbody').find_all('tr'):  # Locate all rows in the table body
        cells = row.find_all('td')  # Data is likely in <td> tags
        if not cells:
            continue
        row_values = [cell.get_text(strip=True) for cell in cells]
        rows_data.append(row_values)

    # 8. Write to CSV in the 'data' folder
    csv_filename = os.path.join(DATA_FOLDER, output_filename)
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)      # Write dynamically extracted header row
        writer.writerows(rows_data)   # Write data rows

    print(f"Data successfully written to {csv_filename}")

    # 9. Close the WebDriver
    driver.quit()

# List of URLs to scrape
urls = [
    "https://tennisabstract.com/reports/mcp_leaders_serve_men_last52.html",
    "https://tennisabstract.com/reports/mcp_leaders_return_men_last52.html",
    "https://tennisabstract.com/reports/mcp_leaders_rally_men_last52.html",
    "https://tennisabstract.com/reports/mcp_leaders_tactics_men_last52.html"
]

# Scrape each URL and save to a separate CSV file
scrape_tennis_data(urls[0], "serve_leaders.csv")
scrape_tennis_data(urls[1], "return_leaders.csv")
scrape_tennis_data(urls[2], "rally_leaders.csv")
scrape_tennis_data(urls[3], "tactics_leaders.csv")