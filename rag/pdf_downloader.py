import os
import requests


def download_pdf_if_not_exists(pdf_path, url):
    if not os.path.exists(pdf_path):
        print("[INFO] File doesn't exist, downloading...")

        response = requests.get(url)

        if response.status_code == 200:
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print("[INFO] Download complete.")
        else:
            print("[ERROR] Failed to download PDF.")
