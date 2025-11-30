from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

results = model.val(
    data="C:/Users/Damian/Desktop/YOLO/data.yaml",
    conf=0.25,
    iou=0.6,
    save_json=True,
    plots=True
)
