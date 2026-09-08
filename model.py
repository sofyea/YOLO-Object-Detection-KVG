from ultralytics import YOLO

def initialize_yolo_model():
    # Muat naik pretrained model YOLOv8 (Nano version)
    model = YOLO("yolov8n.pt")
    
    print("=== MAKLUMAT SENI BINA MODEL YOLOV8 ===")
    print(f"Model Name: YOLOv8n (Nano)")
    print(f"Task Type: Object Detection")
    print(f"Number of Classes: {len(model.names)}")
    print("\nSenarai Kategori Objek Terpilih (COCO Dataset):")
    
    # Tunjuk 10 kelas pertama
    for class_id in range(10):
        print(f"  - Class {class_id}: {model.names[class_id]}")
        
    return model

if __name__ == "__main__":
    initialize_yolo_model()