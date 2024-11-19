from ultralytics import YOLO

model = YOLO("yolo11n.pt") 

results = model.train(data="xView.yaml", epochs=100,val=True,batch=8,imgsz=320)
