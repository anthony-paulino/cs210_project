import gdown
import os

# File mapping
FILES = {
    "raw_dataset.csv": "https://drive.google.com/uc?id=15QJIa6AxucXFIwCEITNL_uioh8cEj8Er",
    "clean_collision_data.csv": "https://drive.google.com/uc?id=12YVvDoTXSMhq65jYXiDpLW0pvLY-J8M1",
    "random_forest_model.pkl": "https://drive.google.com/uc?id=1dwqypIt_eLYZM1tM8UR7g7HBuwxy17df"
}

def download_files(file_mapping, destination_folder="data"):
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    for filename, url in file_mapping.items():
        output_path = os.path.join(destination_folder, filename)
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            gdown.download(url, output_path, quiet=False)
            print(f"Downloaded {filename} to {output_path}.")
        else:
            print(f"{filename} already exists.")

def check_and_download_file(filepath, url):
    """
    Check if the file exists, and if not, download it.
    """
    if not os.path.exists(filepath):
        print(f"{os.path.basename(filepath)} is missing. Downloading now...")
        gdown.download(url, filepath, quiet=False)
        print(f"Downloaded {os.path.basename(filepath)} to {filepath}.")
    else:
        print(f"{os.path.basename(filepath)} is already available at {filepath}.")

if __name__ == "__main__":
    files_data = {k: FILES[k] for i, k in enumerate(FILES) if i < 2} 
    files_models = {k: FILES[k] for i, k in enumerate(FILES) if i >= 2} 
    download_files(files_data) 
    download_files(files_models, "models")
