import re
import time

import pandas as pd
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

ROOT_URL = 'https://pochivka.bg/plazhove-bulgaria-f120'

chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-extensions")

driver = webdriver.Chrome(options = chrome_options) 
def anti_ad(url):    
    driver = webdriver.Chrome(options = chrome_options) 
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
    
    scroll_count = 0
    while scroll_count < 6:   
        driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
        time.sleep(1)   
        scroll_count += 1

    POPUP_ELEMENT = driver.find_element(By.CSS_SELECTOR, "div.fixed-box.quiz") 
    driver.execute_script("arguments[0].remove();", POPUP_ELEMENT)

    
    BACKDROP_ELEMENT = driver.find_element(By.CSS_SELECTOR, 'div.backdrop[style*="display: block;"]')
    driver.execute_script("arguments[0].remove();", BACKDROP_ELEMENT)  
    return driver


ROOT_DRIVER = anti_ad(ROOT_URL)
driver.quit()

ROOT_PAGE_HTML = ROOT_DRIVER.page_source
soup = BeautifulSoup(ROOT_PAGE_HTML, "lxml")

PATTERN = r'span class="map">\s*<img alt="([^"]*)"'
beach_list = re.findall(PATTERN, str(soup))

REMOVE_STRING = ' (плаж)'
beach_list = [item.replace(REMOVE_STRING, '') for item in beach_list]
beach_dict = {'beach_name': beach_list}
TITLE_DIVS = ROOT_DRIVER.find_elements(By.CSS_SELECTOR, "div.title")

url_bank = []
for i, title_div in enumerate(TITLE_DIVS):
    try:
        inner_anchor = title_div.find_element(By.CSS_SELECTOR, "a")
        href_value = inner_anchor.get_attribute("href")
        url_bank.append(href_value)
        
    except:
        break

beach_dict['urls'] = url_bank

latitude_container = []
longitude_container = []

for name_link in beach_dict['urls']:
    try:
        beach_driver = anti_ad(name_link)
        latitude_element = beach_driver.find_element(By.CSS_SELECTOR, 'meta[property="place:location:latitude"]')
        longitude_element = beach_driver.find_element(By.CSS_SELECTOR, 'meta[property="place:location:longitude"]')

        latitude_container.append(latitude_element.get_attribute("content"))
        longitude_container.append(longitude_element.get_attribute("content"))
        driver.quit()
    except TimeoutException:
        driver.quit()
        raise RuntimeError(f"{name_link} is broken or the required elements were not found")

beach_dict.update({'latitude': latitude_container, 'longitude': longitude_container})

beach_info = pd.DataFrame(beach_dict)
beach_info.to_csv('beach_info.csv', index = True)