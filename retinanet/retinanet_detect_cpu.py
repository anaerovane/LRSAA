import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import transforms

def predict_and_draw_boxes(image_path, model, device, threshold=0.3):
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

    plt.show()

image_path = 'C:/Users/Aerovane/Desktop/xml/images/train/5_1.tif'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights='DEFAULT')
model= torchvision.models.detection.retinanet_resnet50_fpn_v2(weights='DEFAULT')
#model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
model.load_state_dict(torch.load('CPU10retinanet.pth'))
model.to(device)
predict_and_draw_boxes(image_path, model, device)