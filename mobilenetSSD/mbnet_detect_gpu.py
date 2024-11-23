import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights

def predict_and_draw_boxes(image_path, model, device, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((320, 320))
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)

    boxes = prediction[0]['boxes'].cpu()
    labels = prediction[0]['labels'].cpu()
    scores = prediction[0]['scores'].cpu()
    print(boxes,labels,scores)

    scale_factor_x = original_width / 320
    scale_factor_y = original_height / 320

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, label, score in zip(boxes, labels, scores):
        if score > threshold: 
            box_scaled = [
                box[0].item() * scale_factor_x,
                box[1].item() * scale_factor_y,
                box[2].item() * scale_factor_x,
                box[3].item() * scale_factor_y
            ]
            rect = patches.Rectangle((box_scaled[0], box_scaled[1]),
                                    box_scaled[2] - box_scaled[0],
                                    box_scaled[3] - box_scaled[1],
                                    linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(box_scaled[0], box_scaled[1], str(label.item()), color='w', backgroundcolor='r')
    plt.savefig('./xiamen/mbnet_2514.png')
    plt.show()

image_path = './xiamen/crop_2514.tif'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
model.load_state_dict(torch.load('GPU10ssdlite320_mobilenet_v3_large.pth'))
model.to(device)
predict_and_draw_boxes(image_path, model, device,threshold=0.2)
