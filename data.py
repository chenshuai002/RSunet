import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F

def custom_collate_fn(batch):

    images, targets = zip(*batch)
    original_sizes = [img.shape[-2:] for img in images]
    max_size = tuple(max(sizes) for sizes in zip(*original_sizes))

    padded_images = []
    padded_targets = []
    for img, target in zip(images, targets):
        pad_h = max_size[0] - img.shape[1]
        pad_w = max_size[1] - img.shape[2]
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), value=0)
            target = F.pad(target, (0, pad_w, 0, pad_h), value=0)
        padded_images.append(img)
        padded_targets.append(target)

    padded_images = [img.clone().detach() for img in padded_images]
    padded_targets = [target.clone().detach() for target in padded_targets]

    padded_images = torch.stack(padded_images)
    padded_targets = torch.stack(padded_targets)

    return padded_images, padded_targets

class CustomDataset(Dataset):
    def __init__(self, image_folder, target_folder, transform=None):
        self.image_folder = image_folder
        self.target_folder = target_folder
        self.transform = transform

        self.image_filenames = os.listdir(image_folder)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_folder, img_name)
        target_path = os.path.join(self.target_folder, img_name)

        image = Image.open(image_path).convert('RGB')
        target = Image.open(target_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return image, target

def SteelDataset(image_dir, target_dir, num_workers=12, batch_size=32, validation_split=0.2, shuffle_dataset=True, random_seed=42):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    custom_dataset = CustomDataset(image_folder=image_dir, target_folder=target_dir, transform=transform)

    dataset_size = len(custom_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(random_seed))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_dataset, num_workers=num_workers, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=custom_collate_fn)

    return train_loader, val_loader