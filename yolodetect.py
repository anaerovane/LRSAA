from ultralytics import YOLO
import os
from tqdm import tqdm

def label_main(output_directory,output_label_dir,output_labelnew_dir,jpg_dir):
    for filename in tqdm(os.listdir(output_label_dir)):
        base,ext=os.path.splitext(filename)
        label_path = os.path.join(output_label_dir,filename)
        label_output_path = os.path.join(output_labelnew_dir,filename)
        tif_path = os.path.join(output_directory,base+".tif")
        jpg_path = os.path.join(jpg_dir,base+".jpg")
        yolo_detect(tif_path,jpg_path,label_path,label_output_path)

def yolo_detect(tif_path,jpg_path,label_path,label_output_path):
    model = YOLO("./yolo_best.pt")  
    results = model([tif_path])  
    with open(label_path, 'r') as f:
        lines = f.readlines()
    coordinates = [float(line.strip()) for line in lines]
    left,upper,right,lower=coordinates[0],coordinates[1],coordinates[2],coordinates[3]
    for result in results:
        boxes = result.boxes  
        masks = result.masks  
        keypoints = result.keypoints  
        probs = result.probs  
        obb = result.obb  
        #result.show()  
        result.save(filename=jpg_path)
        with open(label_output_path, 'w') as ff:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls.item())
                prob = float(box.conf.item())
                newl=x1+left
                newu=y1+upper
                newr=x2+left
                newd=y2+upper
                ff.write(str(newl)+","+str(newu)+","+str(newr)+","+str(newd)+","+str(prob)+","+str(cls)+"\n")


