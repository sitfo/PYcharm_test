import os
import sys
import time

import requests
import langid
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Create a session
session = requests.Session()

# Set the retry parameters
retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[ 500, 502, 503, 504 ])

# Mount it for both http and https usage
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

OUTPUT_DIR = "../source_data/"  # Directory to store downloaded books
BASE_URL = "https://www.gutenberg.org"
AUTHOR_PAGE = "/ebooks/author/53"
FILE_EXTENSION = ".txt"


def main():
    """Main entrypoint for scraping and downloading books."""
    try:
        os.makedirs(OUTPUT_DIR)
    except OSError:
        pass

    next_page = AUTHOR_PAGE

    while next_page:
        response = requests.get(BASE_URL + next_page)
        print(f"Status code: {response.status_code}")
        if response.status_code != 200:
            print("Failed to download the author page.")
            return

        soup = BeautifulSoup(response.content, "lxml")

        books = soup.select('li.booklink')
        if not books:
            print("No books found on the author page.")
            return

        for i, book in enumerate(books, start=1):
            title = book.select_one('.title').get_text(strip=True)
            link = book.select_one('a.link')['href']

            # Check if the title is in English
            lang, _ = langid.classify(title)
            if lang != 'en':
                print(f"Skipping Book {i}: {title} (not in English)")
                continue

            # Skip if the title contains "letter" or "speech"
            if "letter" in title.lower() or "speech" in title.lower():
                print(f"Skipping Book {i}: {title} (contains 'letters' or 'speech')")
                continue

            print(f"Book {i}: {title}, Link: {link}")

            filename = OUTPUT_DIR + title + FILE_EXTENSION
            if os.path.isfile(filename):
                print(f"Book '{title}' already exists in the output directory.")
                continue

            # Navigate to the book's page and find the text download link
            book_page_response = requests.get(BASE_URL + link)
            book_page_soup = BeautifulSoup(book_page_response.content, "lxml")
            text_link_element = book_page_soup.find('a', string='Plain Text UTF-8')

            if text_link_element is None:
                print(f"Skipping Book {i}: {title} (Plain Text UTF-8 link not found)")
                continue

            text_link = text_link_element.get('href')

            # Modify the dl_link to point to the actual text file of the book
            dl_link = "https://www.gutenberg.org" + text_link

            print(f"Downloading Book {i}: {title}")
            start_time = time.perf_counter()

            with open(filename, "wb") as output_file:
                response = requests.get(dl_link, stream=True)
                total_length = int(response.headers.get('Content-Length'))

                if total_length != -1:
                    progress_bar_size = 50
                    chunk_size = total_length // progress_bar_size

                    for data in response.iter_content(chunk_size=chunk_size):
                        output_file.write(data)
                        progress_bar_count = len(data) * progress_bar_size / total_length
                        percentage_download = ((len(data) * progress_bar_size) / total_length) * 100
                        loading_display = ('█' * int(progress_bar_count)) + \
                                          ('.' * (progress_bar_size - int(progress_bar_count)))
                        sys.stdout.write("\r%s %d%%" % (loading_display, percentage_download))
                        sys.stdout.flush()

            end_time = time.perf_counter()
            runtime = end_time - start_time
            size_in_mb = round((os.stat(filename).st_size / (10 ** 6)), 2)
            print(f'\nCompleted Download in {runtime:0.4f} seconds ({size_in_mb} MB)\n')

        next_link = soup.select_one('a[title="Go to the next page of results."]')

        if next_link:
            next_page = next_link['href']
        else:
            next_page = None
