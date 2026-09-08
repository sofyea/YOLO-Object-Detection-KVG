import os
import requests
import cv2

def download_sample_images():
    os.makedirs("./data", exist_ok=True)
    images = {
        "test_desk.jpg": "https://ultralytics.com/images/bus.jpg",
        "test_objects.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    }
    
    for filename, url in images.items():
        filepath = os.path.join("./data", filename)
        if not os.path.exists(filepath):
            print(f"Muat turun {filename}...")
            response = requests.get(url)
            with open(filepath, "wb") as f:
                f.write(response.content)
    print("✓ Dataset dan gambar ujian sedia dalam folder ./data")

if __name__ == "__main__":
    download_sample_images()