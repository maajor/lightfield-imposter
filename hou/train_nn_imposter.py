import torch
torch.autograd.set_detect_anomaly(True)
from torch import nn, optim
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data
import tqdm

# Model
class ImposterNN(nn.Module):
    def __init__(self, D=8, W=256, input_ch=2, skips=[4], RES=256):
        """ 
        """
        super(ImposterNN, self).__init__()
        self.D = D
        self.W = W
        self.RES = RES
        self.input_ch = input_ch
        self.skips = skips

        self.fc1 = nn.Linear(input_ch, W)

        self.fc2 = nn.Linear(W, 2*W)

        self.fc3 = nn.Linear(2*W, 4*W)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(4, 8, 3, stride=1, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(16, 4, 3, stride=1, padding=1)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x).reshape((-1, 4, 16, 16))

        x = self.up1(x)
        x = self.conv1(x)
        x = F.relu(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = self.up3(x)
        x = self.conv3(x)
        x = F.relu(x)

        x = self.up4(x)
        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)

        return x.permute(0,2,3,1)

def collect_images():
    grid=16
    theta_step = np.pi / grid
    phi_step = np.pi * 2 / grid
    im = np.array(Image.open('export/impostor_sh_basecolour.png'))
    w_res = int(im.shape[0]/grid)
    h_res = int(im.shape[1]/grid)
    imgs = np.zeros((grid*grid, w_res,h_res, 4))
    angles = np.zeros((grid*grid, 2))
    imgid = 0
    for i in range(grid): # row
        for j in range(grid): # column
            theta = -np.pi * 0.5 + theta_step/2 + theta_step * j
            phi = -np.pi + phi_step/2 + phi_step * i
            imobj = im[i*w_res:(i+1)*w_res,j*h_res:(j+1)*h_res,:]
            imgs[imgid,:,:,:] = imobj / 127.5 - 1.0
            angles[imgid,0] = theta / ( np.pi )
            angles[imgid,1] = phi / (np.pi * 2)
            imgid += 1
    np.save("dataset/pig_imposter", {"colors":imgs, "coords":angles})


def prepare_imposter_train_dataloader(path="dataset/pig_imposter.npy", batch_size=10):
    dataset = np.load(path, allow_pickle=True)
    colors = dataset.item().get('colors')
    coords = dataset.item().get('coords')

    train_colors = colors
    test_colors = colors[0:colors.shape[0]:8,:,:,:]

    train_coords = coords
    test_coords = coords[0:colors.shape[0]:8,:]

    def loader(color, coord, batch_size):
        color = torch.from_numpy(color).float() 
        coord = torch.from_numpy(coord).float() 

        data_set = Data.TensorDataset(color, coord)

        loader = Data.DataLoader(
            dataset=data_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        return loader

    return loader(train_colors, train_coords, batch_size), loader(test_colors, test_coords, batch_size)

collect_images()

device = torch.device("cuda:0")
BATCH_SIZE = 10
TRAIN_EPOCHS = 1000

train_loader, test_loader = prepare_imposter_train_dataloader(batch_size=BATCH_SIZE)

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
    train_loss = train_loss/ len(train_loader.dataset)
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

    test_loss /= len(test_loader.dataset)
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
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-3, betas=(0.9, 0.999))
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
    model = ImposterNN()
    train(model, "imposternn")

if __name__ == "__main__":
    train_all()