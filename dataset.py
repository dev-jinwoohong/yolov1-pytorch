import torch
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import xmltodict
from imutils import paths
from utils import xyxy_to_xywh

import matplotlib.pyplot as plt
import matplotlib.patches as patches

categories = ["aeroplane", "bicycle", "bird", "boat", "bottle",
              "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]


class VOCDataset(Dataset):
    def __init__(self, root_dir="", image="", annotation="", s=7, b=2, c=20, img_size=448, is_debug=False):
        self.img_dir = os.path.join(root_dir, image)
        self.ano_dir = os.path.join(root_dir, annotation)
        self.S = s
        self.B = b
        self.C = c
        self.img_size = img_size
        self.is_debug = is_debug

        self.transform = transforms.Compose(
            [transforms.Resize((self.img_size, self.img_size)), transforms.ToTensor(), ])

        self.img_paths = list(paths.list_images(self.img_dir))
        self.ano_paths = list(paths.list_files(self.ano_dir))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.img_paths[index]))
        label_matrix = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        with open(self.ano_paths[index], 'r', encoding='utf-8') as file:
            data = xmltodict.parse(file.read())

        objects = data['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]
        original_height = float(data['annotation']['size']['height'])
        original_width = float(data['annotation']['size']['width'])

        for obj in objects:
            obj_name = obj['name']
            bndbox = obj['bndbox']
            class_name = categories.index(obj_name)
            x, y, w, h = xyxy_to_xywh(
                [float(bndbox['xmin']), float(bndbox['ymin']), float(bndbox['xmax']), float(bndbox['ymax'])],
                original_width, original_height, img_size=self.img_size)

            i, j = int(self.S * y), int(self.S * x)

            x_cell, y_cell = self.S * x - j, self.S * y - i

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1

                box_coordinates = torch.tensor(
                    [x_cell, y_cell, w, h]
                )

                label_matrix[i, j, 21:25] = box_coordinates

                label_matrix[i, j, class_name] = 1

        if self.is_debug is True:
            draw_bounding_boxes(image, label_matrix)

        return image, label_matrix


def draw_bounding_boxes(image, label_matrix, s=7, img_size=448):
    image = image.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for i in range(s):
        for j in range(s):
            cell_label = label_matrix[i, j]
            if cell_label[20] == 1:
                x, y, w, h = cell_label[21:25]
                cx = img_size / s * (j + x)
                cy = img_size / s * (i + y)
                weight = w * img_size
                height = h * img_size
                rect = patches.Rectangle(
                    (cx - weight / 2, cy - height / 2),
                    weight,
                    height,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none"
                )
                ax.add_patch(rect)

                class_id = cell_label[0:20].argmax()
                class_name = categories[class_id]
                ax.text(cx - weight / 2, cy - height / 2, class_name, color='white', backgroundcolor='red', fontsize=10,
                        verticalalignment='top')

    plt.show()


if __name__ == "__main__":
    train_dataset = VOCDataset(root_dir=r"C:\Users\X299\Downloads\VOCdevkit_train\VOC2007", image="JPEGImages",
                               annotation="Annotations", is_debug=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, sampler=None,
    )

    for idx, (images, labels) in enumerate(train_loader):
        break
