import os
import zipfile


def safe_extract_zip(zip_path, extract_path="data/"):
    try:
        os.makedirs(extract_path, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                try:
                    zip_ref.extract(member, path=extract_path)
                    print(f"Extracted: {member}")
                except Exception as e:
                    print(f"Error extracting {member}: {e}")

        print("Extraction completed")
    except Exception as e:
        print(f"Extraction failed: {e}")


if __name__ == "__main__":
    # Place the zip file inside the data/ folder once you get it, then run this script
    safe_extract_zip("data/Alzheimer-s Disease Detection.v1i.yolov11.zip", "data/")