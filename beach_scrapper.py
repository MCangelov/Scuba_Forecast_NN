import re
import time

import pandas as pd
from typing import List

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


def navigate_to_url(driver: WebDriver, url: str, wait_time: int = 10) -> None:
    """
    Navigates the webdriver to a specified URL and waits until the page is loaded.

    Parameters:
    driver (webdriver): The webdriver instance to use.
    url (str): The URL to navigate to.
    wait_time (int, optional): The maximum time to wait for the page to load. Defaults to 10.

    """
    try:
        driver.get(url)
        WebDriverWait(driver, wait_time).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body')))
    except TimeoutException:
        raise RuntimeError(
            f"Timeout error while navigating to {url}. Page did not load successfully.")
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while navigating to {url}: {str(e)}")


def scroll_to_bottom(driver: WebDriver) -> None:
    """
    Scrolls the webdriver to the bottom of the page.
    Will keep scrolling down the page until it cannot scroll any further.

    Parameters:
    driver (webdriver): The webdriver instance to use.

    """
    initial_page_source = driver.page_source
    can_scroll = True
    while can_scroll:
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        page_source_after_scroll = driver.page_source
        if initial_page_source == page_source_after_scroll:
            can_scroll = False
        else:
            initial_page_source = page_source_after_scroll


def remove_elements(driver: WebDriver, css_selectors: List[str]) -> None:
    """
    Removes elements from the webpage that match any of the provided CSS selectors.

    Parameters:
    driver (webdriver): The webdriver instance to use.
    css_selectors (list[str]): A list of CSS selectors. Each element that matches any of these selectors will be removed.

    """
    for css_selector in css_selectors:
        try:
            element = driver.find_element(By.CSS_SELECTOR, css_selector)
            driver.execute_script("arguments[0].remove();", element)
        except (NoSuchElementException, TimeoutException):
            pass


ROOT_URL = 'https://pochivka.bg/plazhove-bulgaria-f120'
CSS_ELEMENTS = ["div.fixed-box.quiz", 'div.backdrop[style*="display: block;"]']
PATTERN = r'span class="map">\s*<img alt="([^"]*)"'
REMOVE_STRING = ' (плаж)'

# Options are set for driver to run in the background
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-extensions")
driver = webdriver.Chrome(options=chrome_options)

navigate_to_url(driver, ROOT_URL)
scroll_to_bottom(driver)
remove_elements(driver, CSS_ELEMENTS)

soup = BeautifulSoup(driver.page_source, "lxml")
beach_list = re.findall(PATTERN, str(soup))
beach_list = [item.replace(REMOVE_STRING, '') for item in beach_list]
beach_dict = {'beach_name': beach_list}
# Needed to locate the beach URLs in the HTML
title_divs = driver.find_elements(By.CSS_SELECTOR, "div.title")

# Gathering the URLs of every beach
url_storage = []
for i, title_div in enumerate(title_divs):
    try:
        inner_anchor = title_div.find_element(By.CSS_SELECTOR, "a")
        href_value = inner_anchor.get_attribute("href")
        url_storage.append(href_value)

    except NoSuchElementException:
        break

beach_dict['urls'] = url_storage

latitude_container = []
longitude_container = []

# Extracting the latitude and longitude of every beach
try:
    for name_link in beach_dict['urls']:
        try:
            # Have to redo for every new link we visit
            navigate_to_url(driver, name_link)
            scroll_to_bottom(driver)
            remove_elements(driver, CSS_ELEMENTS)

            latitude_element = driver.find_element(
                By.CSS_SELECTOR, 'meta[property="place:location:latitude"]')
            longitude_element = driver.find_element(
                By.CSS_SELECTOR, 'meta[property="place:location:longitude"]')

            latitude_container.append(
                latitude_element.get_attribute("content"))
            longitude_container.append(
                longitude_element.get_attribute("content"))
        except TimeoutException:
            raise RuntimeError(
                f"Timeout error while processing {name_link}. The required elements were not found.")
finally:
    driver.quit()

beach_dict.update({'latitude': latitude_container,
                  'longitude': longitude_container})

beach_info = pd.DataFrame(beach_dict)
beach_info.to_csv('beach_info.csv', index=True)
