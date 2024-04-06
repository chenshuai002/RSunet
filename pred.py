import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import DenseResUnetPlus
from PIL import Image
import torch.nn.functional as F

def custom_collate_fn(batch):
    images = batch

    original_sizes = [img.shape[-2:] for img in images]

    max_size = tuple(max(sizes) for sizes in zip(*original_sizes))

    padded_images = []
    for img in images:
        pad_h = max_size[0] - img.shape[1]
        pad_w = max_size[1] - img.shape[2]
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0)
        padded_images.append(img)

    padded_images = [img.clone().detach() for img in padded_images]
    padded_images = torch.stack(padded_images)

    return padded_images


class CustomDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        self.image_filenames = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_folder, img_name)

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def predict(net, data_folder, save_folder):
    net.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    dataset = CustomDataset(image_folder=data_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=custom_collate_fn)

    os.makedirs(save_folder, exist_ok=True)

    with torch.no_grad():
        for i, images in enumerate(dataloader):
            outputs = net(images)

            for j, pred in enumerate(outputs):
                pred = pred.permute(1, 2, 0).cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_image = Image.fromarray(pred)
                pred_image.save(os.path.join(save_folder, f'pred_{i}_{j}.png'))

if __name__ == '__main__':
    model = DenseResUnetPlus(channel=3, num_class=3)
    model.load_state_dict(torch.load('model2.pth'))

    data_folder = 'images/test/sel'
    save_folder = 'images/test/pred_sel'

    predict(model, data_folder, save_folder)