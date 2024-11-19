import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.transforms import transforms
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

class CustomDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.tif')]

        print(f"Found {len(self.image_files)} image files in {image_folder}")
        image_file2=[]
        for image_file in tqdm(self.image_files):
            label_file = os.path.join(label_folder, image_file.replace('.tif', '.txt'))
            if not os.path.exists(label_file):
                print(f"Warning: Label file not found for {image_file}")
            else:
                image_file2.append(image_file)
        self.image_files=image_file2
        print(f"Detecting {len(self.image_files)} image files in {image_folder}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        label_path = os.path.join(self.label_folder, image_file.replace('.tif', '.txt'))

        #print(f"Loading image: {image_path}")
        #print(f"Loading label: {label_path}")

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        with open(label_path, 'r') as f:
            lines = f.readlines()
            boxes = []
            labels = []
            for line in lines:
                parts = line.strip().split(' ')
                if len(parts) != 5:
                    raise ValueError(f"Invalid label format in {label_path}: {line}")
                class_id = int(parts[0])
                center_x = float(parts[1]) * width
                center_y = float(parts[2]) * height
                width_box = float(parts[3]) * width
                height_box = float(parts[4]) * height
                x_min = center_x - width_box / 2
                y_min = center_y - height_box / 2
                x_max = center_x + width_box / 2
                y_max = center_y + height_box / 2
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)
                else:
                    print(f"Warning: Invalid box found in {label_path}: {line}")

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transform:
            image = self.transform(image)

        return image, target

def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    images = torch.stack(images)
    boxes = [target['boxes'] for target in targets]
    labels = [target['labels'] for target in targets]

    boxes_padded = pad_sequence(boxes, batch_first=True, padding_value=0.0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1) 

    targets_padded = []
    for i in range(len(targets)):
        target_padded = {
            'boxes': boxes_padded[i],
            'labels': labels_padded[i]
        }
        targets_padded.append(target_padded)

    return images, targets_padded


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((320, 320)) 
])

image_folder = 'E:/mbnet/1110/data1110/images/train'
label_folder = 'E:/mbnet/1110/data1110/labels/train'
dataset = CustomDataset(image_folder, label_folder, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0,collate_fn=collate_fn)

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
model.train()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)


num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in tqdm(data_loader):
        
        valid_targets = []
        for target in targets:
            valid_indices = target['labels'] >= 0
            valid_target = {
                'boxes': target['boxes'][valid_indices],
                'labels': target['labels'][valid_indices]
            }
            valid_targets.append(valid_target)

        images = list(image for image in images)
        targets = valid_targets

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

torch.save(model.state_dict(), 'CPUssdlite320_mobilenet_v3_large.pth')