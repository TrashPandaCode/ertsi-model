from webdav3.client import Client
import os
import zipfile
import shutil

# https://th-koeln.sciebo.de/s/AMAu4NWnXfz1Go6

# Configuration for public ownCloud link
options = {
    "webdav_hostname": "https://th-koeln.sciebo.de/public.php/webdav/",
    "webdav_login": "",  # Leave empty for public link
    "webdav_password": "",  # Leave empty for public link
    "root": "/",  # Root path of the shared folder
}
client = Client(options)

# Public token from the link
client.session.auth = ("AMAu4NWnXfz1Go6", "")  # Use token as username, empty password

remote_path = "/data.zip"
local_path = "./data.zip"

hash = client.info(remote_path)["etag"]

if (
    not os.path.exists("data_version.txt")
    or hash != open("data_version.txt").read().strip()
):
    print("Downloading new data...")
    client.download_sync(remote_path=remote_path, local_path=local_path)
    with open("data_version.txt", "w") as f:
        f.write(hash)
else:
    print("Data is up to date.")
    exit()

# Unzip the downloaded file
with zipfile.ZipFile(local_path, "r") as zip_ref:
    # delte the old data
    if os.path.exists("data"):
        shutil.rmtree("data")
    
    zip_ref.extractall(".")
    print("Data unzipped successfully.")

    # Remove the zip file after extraction
    os.remove(local_path)
    print("Zip file removed.")
