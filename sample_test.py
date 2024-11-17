
from torchvision.utils import draw_bounding_boxes
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
import h5py
import io
import numpy as np
import matplotlib.pyplot as plt
import random
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import xmltodict
import copy
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



class CustomDataset(Dataset):

    item_dict = {
        'hat' : 1,
        'sunglass' : 2,
        'jacket' : 3,
        'shirt' : 4,
        'pants' : 5,
        'shorts' : 6,
        'skirt' : 7,
        'dress' : 8,
        'bag' : 9,
        'shoe' : 10
    }

    def __init__(self, image_dir, annos_dir, image_list, annos_list, transforms=None):
        self.transforms = transforms
        self.images = image_list
        self.annos = annos_list
        self.image_dir = image_dir
        self.annos_dir = annos_dir

    def __getitem__(self, idx):

        image_label = self.images[idx]
        image_path = os.path.join(self.image_dir, image_label)
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        annos_label = self.annos[idx]

        boxes = []
        labels = []

        with open(annos_label, 'r') as f:
            dict_curr = xmltodict.parse(f.read(), force_list=('object',))

        for dict_obj in dict_curr['annotation']['object']:
           
            if 'bndbox' in dict_obj:
                xmin = int(dict_obj['bndbox']['xmin'])
                ymin = int(dict_obj['bndbox']['ymin'])
                xmax = int(dict_obj['bndbox']['xmax'])
                ymax = int(dict_obj['bndbox']['ymax'])
                boxes.extend([xmin, ymin, xmax, ymax])
     
            if 'name' in dict_obj:
                label_name = dict_obj['name']

                if label_name in CustomDataset.item_dict:
                    labels.append(CustomDataset.item_dict[label_name])

        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)

        if self.transforms is not None:
            transformed = self.transforms(image = image, bboxes = boxes, labels = labels)

        image = torch.as_tensor(transformed['image'], dtype = torch.float32) / 255.0
        boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

  

        return image, target

    def __len__(self):
        return len(self.images)
    
def get_transform(train):
    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        transform = A.Compose([
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    return transform


def grab_sample(image_path, annos_path, num_samples):


    image_list = [image for image in os.listdir(image_path)]
    selected_images = random.sample(image_list, num_samples)
    selected_annos = []
    for img in selected_images:
        base_name = os.path.splitext(img)[0]
        anno_filename = f"{base_name}.xml"
        anno_filepath = os.path.join(annos_path, anno_filename)
        selected_annos.append(anno_filepath)

    return selected_images, selected_annos


image_dir = "colorful_fashion_dataset_for_object_detection/JPEGImages"
annos_dir = "colorful_fashion_dataset_for_object_detection/Annotations"

image_samples, anno_samples = grab_sample(image_dir, annos_dir, 16)

test_dataset = CustomDataset(image_dir, annos_dir ,image_samples, anno_samples, transforms=get_transform(train=True))

sample_loader = DataLoader(test_dataset, batch_size= 4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers= 1 )


def plot_batch(images, targets, class_names=None, batch_size=4, cols=2):
    """
    Plots a batch of images with their corresponding bounding boxes and labels.

    Parameters:
    - images (list of torch.Tensor): List of image tensors with shape (C, H, W) and values in [0, 1].
    - targets (list of dict): List of target dictionaries containing 'boxes' and 'labels'.
    - class_names (dict, optional): Dictionary mapping label indices to class names.
    - batch_size (int, optional): Number of images to plot from the batch. Default is 4.
    - cols (int, optional): Number of columns in the plot grid. Default is 2.
    """
    # Determine the number of images to plot
    batch_size = min(batch_size, len(images))
    rows = (batch_size + cols - 1) // cols  # Ceiling division

    # Create a matplotlib figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))
    axes = axes.flatten()  # Flatten in case of multiple rows

    for idx in range(batch_size):
        image = images[idx].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        target = targets[idx]
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()

        ax = axes[idx]
        ax.imshow(image)
        ax.axis('off')

        # Plot each bounding box
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            # Denormalize coordinates
            # Calculate width and height of the bounding box
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min

            # Create a Rectangle patch
            rect = patches.Rectangle(
                (x_min, y_min),
                bbox_width,
                bbox_height,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )

            # Add the rectangle to the plot
            ax.add_patch(rect)

            # Add label if provided
            if class_names is not None:
                label_text = class_names.get(label, str(label))
            else:
                label_text = str(label)

            # Add text label above the bounding box
            ax.text(
                x_min,
                y_min - 10,  # Slightly above the box
                label_text,
                fontsize=12,
                color='yellow',
                backgroundcolor='red'
            )

    # Hide any remaining subplots if batch_size < rows * cols
    for idx in range(batch_size, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('output.png') 

image, targets = next(iter(sample_loader))    

plot_batch(image, targets)