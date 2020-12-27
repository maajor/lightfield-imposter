import os
import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
import tqdm

from utils.loader import collect_3d_imposter_images, prepare_3d_imposter_train_dataloader
from utils.nets import ImposterNN

RES = 128
collect_3d_imposter_images(grid=16, filename='export/impostor_sh_N.png', frame_res=RES)

device = torch.device("cuda:0")
BATCH_SIZE = 10
TRAIN_EPOCHS = 500

train_loader, test_loader = prepare_3d_imposter_train_dataloader(batch_size=BATCH_SIZE)

loss_fn = torch.nn.MSELoss(reduction='sum')

def train_epoch(epoch, model, optimizer):
    model.train()
    train_loss = 0
    for id, (colors, coords) in enumerate(train_loader):
        coords = coords.to(device)
        optimizer.zero_grad()
        pred_colors = model(coords)
        colors = colors.to(device)
        loss = loss_fn(pred_colors, colors)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss = train_loss * 255 / (len(train_loader.dataset) * (RES*RES*4))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss))
    return train_loss

def test_epoch(epoch, model):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for id, (colors, coords) in enumerate(test_loader):
            coords = coords.to(device)
            pred_colors = model(coords)
            colors = colors.to(device)
            test_loss += loss_fn(pred_colors, colors).item()

    test_loss = test_loss * 255 / (len(test_loader.dataset) * (RES*RES*4))
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss

def plot(train_loss, test_loss, save_name):
    iter_range = range(TRAIN_EPOCHS)
    plt.subplot(2, 1, 1)
    plt.plot(iter_range, np.log10(train_loss), 'o-')
    plt.title('Train Loss vs. Epoches')
    plt.ylabel('Train Loss')
    plt.subplot(2, 1, 2)
    plt.plot(iter_range[int(len(test_loss)/2):], np.log10(np.array(test_loss))[int(len(test_loss)/2):], '.-')
    plt.xlabel('Log Test Loss vs. Epoches')
    plt.ylabel('Log Test Loss Last Half')
    plt.savefig(save_name+".png")
    plt.close()

def train(model, save_name):
    train_loss = []
    test_loss = []
    model_path = 'model/' + save_name + ".pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load('model/' + save_name + ".pth"))
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999))
    loop = tqdm.tqdm(range(TRAIN_EPOCHS))
    for epoch in loop:
        train_loss.append(train_epoch(epoch, model, optimizer))
        test_loss.append(test_epoch(epoch, model))
    torch.save(model.state_dict(), "model/" + save_name + ".pth")
    plot(train_loss, test_loss, save_name)

def test(model, saved_name):
    model.load_state_dict(torch.load('model/' + saved_name + ".pth"))
    model = model.to(device)
    model.eval()
    test_epoch(0, model)

def train_all():
    model = ImposterNN(RES=RES)
    train(model, "imposternn")

if __name__ == "__main__":
    train_all()