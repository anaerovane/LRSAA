import torch
from PIL import Image
from torchvision.models.detection.ssdlite import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import torchvision
import matplotlib.patches as patches

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=torchvision.models.detection.ssd300_vgg16(weights='DEFAULT')
model.load_state_dict(torch.load('GPU10ssdvgg.pth'))
model.to(device)
model.eval()  

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = F.resize(image, (320, 320))
    image = F.to_tensor(image)
    return image.unsqueeze(0).to(device)  

def plot_prediction(image, prediction, threshold=0.3):
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    for box, label, score in zip(boxes, labels, scores):
        if score > threshold:  
            print(box,label,score)
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], str(label), color='white', verticalalignment='top', bbox={'color': 'red', 'pad': 0})
    plt.show()

image_path = '/mnt/data1/1110/data1116/images/train/5_6.tif'
input_image = preprocess_image(image_path)

with torch.no_grad():
    predictions = model(input_image)

prediction = predictions[0]

original_image = Image.open(image_path).convert("RGB")
original_image = original_image.resize((320, 320))
plot_prediction(original_image, prediction)

