import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils.loader import collect_3d_imposter_images, prepare_3d_imposter_train_dataloader
from utils.nets import ImposterNN

# collect_images()

device = torch.device("cuda:0")
TRAIN_EPOCHS = 300

train_loader, test_loader = prepare_3d_imposter_train_dataloader(batch_size=1)

loss_fn = torch.nn.MSELoss(reduction='sum')


def render_by_angle(model, theta, phi):
    with torch.no_grad():
        coords = torch.Tensor([theta, phi]).to(device)
        pred_colors = model(coords.unsqueeze(0))
    pred_colors = pred_colors.cpu()
    return pred_colors

def plot_compare(saved_name):
    model = ImposterNN()
    model.load_state_dict(torch.load('model/' + saved_name + ".pth"))
    model = model.to(device)
    model.eval()

    dataset = np.load("dataset/3d_imposter.npy", allow_pickle=True)
    colors = dataset.item().get('colors')
    coords = dataset.item().get('coords')

    GRID = 16
    theta = 0.4
    phi_min = -0.3
    phi_max = 0.3
    phi_step = 0.01
    imgnum = int((phi_max-phi_min)/phi_step)
    grid_j = int((theta+0.5+0.5/GRID)/(1/GRID))-1
    for i in range(imgnum):
        phi = phi_min+i*phi_step
        grid_i = int((phi+0.5+0.5/GRID)/(1/GRID))-1
        original_id = grid_i*GRID+grid_j
        pred_colors = render_by_angle(model, theta, phi)
        color = colors[original_id,:,:,:]

        _, axs = plt.subplots(nrows=1, ncols=2)
        axs[0].set_title("Original")
        axs[0].imshow(color, cmap='gray')
        axs[1].set_title("Prediction")
        axs[1].imshow(pred_colors[0,:,:,:], cmap='gray')
        plt.savefig("tmp/compare_nn_imposter_{0:03d}".format(i))
        plt.close()
    #colors, coords = next(iter(train_loader))
    #plt.show()

def gen_gif():
    imgs=[]
    for f in os.listdir("tmp"):
        im =Image.open('tmp/{0}'.format(f))
        imgs.append(im)
    imgs[0].save('compare_nn_imposter.gif',
               save_all=True, append_images=imgs[1:], optimize=True, loop=0)

if __name__ == "__main__":
    plot_compare("imposternn")
    gen_gif()