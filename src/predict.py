import os
from PIL import Image
from ultralytics import YOLO

# Load the trained model from local models/ folder
model = YOLO("models/best.pt")

VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp")


def predict_folder_grayscale(
    image_folder: str = "data/test/images",
    save_dir: str = "outputs/predicted_images",
):
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    os.makedirs(save_dir, exist_ok=True)

    image_files = [
        f
        for f in os.listdir(image_folder)
        if os.path.isfile(os.path.join(image_folder, f))
        and f.lower().endswith(VALID_EXTENSIONS)
    ]

    print(f"Found {len(image_files)} images in {image_folder}...")

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        base_name = os.path.splitext(image_file)[0]

        # Convert the image to grayscale
        with Image.open(image_path) as img:
            gray_image = img.convert("L")
            gray_image_path = os.path.join(save_dir, f"{base_name}_gray.jpg")
            gray_image.save(gray_image_path)

        # Perform prediction using the YOLO model and save inside save_dir
        results = model.predict(
            source=gray_image_path,
            save=True,
            project=save_dir,
            name=base_name,
            exist_ok=True,
        )

    print(f"All predictions saved inside {save_dir}/")


if __name__ == "__main__":
    predict_folder_grayscale()