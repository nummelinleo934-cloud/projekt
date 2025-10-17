from ultralytics import YOLO

model = YOLO("yolo11n.pt")

result = model(r"C:\Users\leo.nummelin\projekt\object_detection",
               conf=0.5,
               line_width=5,
               save=True,
               project=r"C:\Users\leo.nummelin\projekt\runs",
               name="detect")

boxes = result.boxes
masks = result.masks
keypoints = result.keypoints
probs = result.probs
obb = result.obb