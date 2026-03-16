from ultralytics import YOLO
yolo = YOLO(model='yolo11l.pt',task='predict')
result = yolo(source=0,save=True)