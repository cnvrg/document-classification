import urllib
import requests
from bs4 import BeautifulSoup
from extractor import text_extraction

extracting = text_extraction()


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


# function to check if the link is valid url pointing to a pdf or txt
# it wil read pdf and txt files from a valid url and return a exception otherwise
def extract_data(download_url, filename="tt"):
    try:
        # if url is not from google drive and its pdf
        if download_url.find(".pdf") != -1:
            response = urllib.request.urlopen(download_url)
            file = open(filename + ".pdf", "wb")
            file.write(response.read())
            file.close()
            text = extracting.master_extractor(filename + ".pdf")
            return text

        # if url is not from google drive and its txt
        elif download_url.find(".txt") != -1:
            response = urllib.request.urlopen(download_url)
            file = open(filename + ".txt", "wb")
            file.write(response.read())
            file.close()
            text = extracting.master_extractor(filename + ".txt")
            return text

        else:
            # file is from google drive, now getting the extension of that file (.txt or .pdf)
            r = requests.get(download_url)
            soup = BeautifulSoup(r.content, features="lxml")
            for name in soup.findAll("title"):
                link = name.string
                name = link.replace(" - Google Drive", "")
                print(name)

            # if extenions is pdf
            if name.find(".pdf") != -1:
                download_url = download_url.split("/")[5]
                URL = "https://docs.google.com/uc?export=download"

                session = requests.Session()

                response = session.get(URL, params={"id": download_url}, stream=True)
                token = get_confirm_token(response)

                if token:
                    params = {"id": id, "confirm": token}
                    response = session.get(URL, params=params, stream=True)
                filename = filename + ".pdf"
                save_response_content(response, filename)
                text = extracting.master_extractor(filename)
                return text
            else:
                # if extentions is .txt
                download_url = download_url.split("/")[5]
                URL = "https://docs.google.com/uc?export=download"

                session = requests.Session()

                response = session.get(URL, params={"id": download_url}, stream=True)
                token = get_confirm_token(response)

                if token:
                    params = {"id": id, "confirm": token}
                    response = session.get(URL, params=params, stream=True)
                filename = filename + ".txt"
                save_response_content(response, filename)
                text = extracting.master_extractor(filename)
                return text
    except:
        raise Exception("URL not valid")
