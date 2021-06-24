import shutil
import tarfile
import tempfile
from pathlib import Path

import requests

from rxnrep.utils.io import to_path


def download_file_from_google_drive(file_id, destination):
    """
    Take from https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python
    which is from https://stackoverflow.com/a/39225039
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


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


def download_model(file_id: str, date: str, directory: Path):
    """
    Download a shared tarball file of the model from Google Drive with `file_id`,
    untar it and move it to `directory`.

    For info on how to find the file_id, see
    https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99
    """

    with tempfile.TemporaryDirectory() as dirpath:

        fname = "rxnrep_model.tar.gz"
        fname2 = fname.split(".")[0]
        fname = to_path(dirpath).joinpath(fname)
        fname2 = to_path(dirpath).joinpath(fname2)

        print(
            "Start downloading pretrained model from Google Drive; this may take a while."
        )
        download_file_from_google_drive(file_id, fname)

        if not tarfile.is_tarfile(fname):
            model_path = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
            raise RuntimeError(
                f"Failed downloading model from Google Drive. You can try download the "
                f"model manually at: {model_path}, untar it, and pass the path to the "
                f"model to bondnet to use it; i.e. do something like: "
                f"$ bondnet --model <path_to_model> ..."
            )

        tf = tarfile.open(fname, "r:gz")
        tf.extractall(fname2)

        # copy content to the given directory
        # note, we need to joinpath date because the download file from Google Drive
        # with extract to a directory named date
        shutil.copytree(fname2.joinpath(date), to_path(directory))

        print(f"Finish downloading pretrained model; placed at: {to_path(directory)}")
