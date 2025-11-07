from ultralytics import YOLO

model = YOLO("yolo11n.pt")

result = model(r"C:\Users\leo.nummelin\projekt\object_detection",
               conf=0.5, 
               line_width=5,
               save=True,
               project=r"C:\Users\leo.nummelin\projekt\runs",
               name="detect")

for res in result:
    boxes = res.boxes
    masks = res.masks
    keypoints = res.keypoints
    probs = res.probs
    obb = res.obb