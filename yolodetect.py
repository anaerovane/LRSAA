from ultralytics import YOLO

def yolo_detect(img_path,tif_path,jpg_path,label_path,label_output_path):
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
                newl=x1+left
                newu=y1+upper
                newr=x2+left
                newd=y2+upper
                ff.write(str(newl)+","+str(newu)+","+str(newr)+","+str(newd)+"\n")


