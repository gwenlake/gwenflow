from bs4 import BeautifulSoup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# Define a custom extractor function to extract text from HTML using BeautifulSoup
def custom_extractor(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text()

def web_scapper(url:str ) -> str:
    url = "https://www.mietjmu.in/"



    # Instantiate the RecursiveUrlLoader
    loader = RecursiveUrlLoader(url=url, extractor=custom_extractor)

    # Load the data from the website
    docs = loader.load()

    # Define the file path to store the data
    output_file = "scraped__data.txt"

    # Open the file in write mode with UTF-8 encoding
    with open(output_file, "w", encoding="utf-8") as file:
        # Write metadata and content for each document to the file
        for doc in docs:
            title = doc.metadata.get("title")
            source = doc.metadata.get("source")
            content = doc.page_content

            # Ensure that the title, source, and content are string type
            if isinstance(title, str) and isinstance(source, str) and isinstance(content, str):
                file.write("Page Title: " + title + "\n")
                file.write("Page URL: " + source + "\n")
                file.write("Page Content:\n" + content + "\n\n")
            else:
                print("Skipped a document due to non-string content.")

    print("Data has been successfully written to", output_file)