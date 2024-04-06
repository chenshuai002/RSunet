import os
import time
import warnings
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from matplotlib import pyplot as plt
from data import SteelDataset
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_msssim import ssim

warnings.filterwarnings("ignore")
seed = 69
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def provider(
        data_folder,
        df_path,
        phase,
        batch_size=4,
        num_workers=8,
):
    image_dir = data_folder
    target_dir = df_path
    train_iter, test_iter = SteelDataset(image_dir, target_dir, batch_size=batch_size, num_workers=num_workers)
    dataloader = train_iter if phase == 'train' else test_iter
    return dataloader

def metric(probability, truth):
    with torch.no_grad():
        device = probability.device
        probability = probability.to(device)
        truth = truth.to(device)
        assert (probability.shape == truth.shape)

        ssim1 = ssim(probability, truth)
        ssim1 = torch.tensor([[ssim1.item()]])

    return ssim1

class Meter:
    def __init__(self):
        self.base_ssim_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        ssim = metric(probs, targets)
        self.base_ssim_scores.extend(ssim.tolist())

    def get_metrics(self):
        ssim3 = np.nanmean(self.base_ssim_scores)
        return ssim3

def epoch_log(epoch_loss, meter):
    ssim3 = meter.get_metrics()
    print("Loss: %0.4f | SSIM: %0.4f" % (epoch_loss, ssim3))
    return ssim3

class WeightedLoss(torch.nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, prediction, target):
        mse_loss = torch.nn.functional.mse_loss(prediction, target)
        ssim_loss = 1 - ssim(prediction, target)
        alpha = mse_loss / (mse_loss + ssim_loss)
        weighted_loss = alpha * mse_loss + (1 - alpha) * ssim_loss

        return weighted_loss

class Trainer(object):
    def __init__(self, model, pretrained_model_path=None):
        self.num_workers = 18
        self.batch_size = {"train":36, "val":36}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 0.005
        self.num_epochs = 50
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = model.to(self.device)
        self.criterion = WeightedLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, amsgrad=True)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=10, verbose=True)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.ssim_scores = {phase: [] for phase in self.phases}

        if pretrained_model_path:
            self.load_pretrained_model(pretrained_model_path)

    def load_pretrained_model(self, pretrained_model_path):
        self.net.load_state_dict(torch.load(pretrained_model_path))

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter()
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch + 1} | phase: {phase} | ⏰: {start}")
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach()
            meter.update(targets, outputs)
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        ssim = epoch_log(epoch_loss, meter)
        self.losses[phase].append(epoch_loss)
        self.ssim_scores[phase].append(ssim)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), "model.pth")

if __name__ == '__main__':
    train_df_path = 'images/1/sel'
    data_folder = "images/1/target_sel"

    from model import DenseResUnetPlus

    model = DenseResUnetPlus(channel=3, num_class=3)

    # 预训练模型参数路径
    # pretrained_model_path = 'pretrained_model.pth'
    model_trainer = Trainer(model)
    model_trainer.start()

    losses = model_trainer.losses
    ssim_scores = model_trainer.ssim_scores

    def plot(scores, name):
        plt.figure(figsize=(15, 5))
        plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
        plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
        plt.title(f'{name} plot')
        plt.xlabel('Epoch')
        plt.ylabel(f'{name}')
        plt.legend()

    plot(losses, "Loss")
    plt.savefig('Loss.png')

    plot(ssim_scores, "SSIM")
    plt.savefig('SSIM.png')
