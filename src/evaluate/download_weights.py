import os
import requests


def download_model_weights(
    filename: str,
    base_url: str = "https://minio.lab.sspcloud.fr/jpeignon/NLP_3A_ENSAE/",
):
    url = base_url + filename
    os.makedirs("outputs/", exist_ok=True)
    dest_path = os.path.join("outputs/", filename)

    if os.path.exists(dest_path):
        print(f"Skipping {filename} (already exists)")
        return

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")
