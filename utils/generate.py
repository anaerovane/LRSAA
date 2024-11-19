import os
import json
from tqdm import tqdm
from PIL import Image, ImageDraw

def readtxt(output_labelnew_path):
    result=[]
    if output_labelnew_path.endswith('.txt'):
        with open(txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                resultin=dict()
                numbers = line.strip().split(',')
                resultin['left']=float(numbers[0])
                resultin['upper']=float(numbers[1])
                resultin['right']=float(numbers[2])
                resultin['lower']=float(numbers[3])
                resultin['prob']=float(numbers[4])  
                resultin['cate']=float(numbers[5])
                result.append(resultin)
        return result
    
    for obj in tqdm(os.listdir(output_labelnew_path)):
        txt_path = os.path.join(output_labelnew_path, obj)
        resultin=dict()
        with open(txt_path, 'r', encoding='utf-8') as file:
            for line in file:
                resultin=dict()
                numbers = line.strip().split(',')
                resultin['left']=float(numbers[0])
                resultin['upper']=float(numbers[1])
                resultin['right']=float(numbers[2])
                resultin['lower']=float(numbers[3])
                resultin['prob']=float(numbers[4])  
                resultin['cate']=float(numbers[5])
                result.append(resultin)
    return result

def draw_bbox(tif_path, out_tif_path,result):
    Image.MAX_IMAGE_PIXELS = 10000000000  
    image = Image.open(tif_path)
    draw = ImageDraw.Draw(image)

    for rect in tqdm(result):
        left = rect['left']
        top = rect['upper']
        right = rect['right']
        bottom = rect['lower']
        rect_position = (left, top, right, bottom)  
        #print(rect_position)
        draw.rectangle(rect_position, outline="red", width=1)

    image.save(out_tif_path)