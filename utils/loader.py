import os, re
from PIL import Image
import numpy as np
import torch
import torch.utils.data as Data

pattern = re.compile(r"\w*.[-0-9]*.[0-9]*.png")

def parse_image_angle(img_name):
    theta = np.deg2rad(float(img_name.split(".")[1])+90)
    phi = np.deg2rad(float(img_name.split(".")[2]))
    return theta, phi

def collect_images():
    coords = []
    colors = []

    for f in os.listdir("render"):
        m = pattern.match(f)
        if m:
            im = np.array(Image.open('render/{0}'.format(f)))
            theta, phi = parse_image_angle(f)
            coords.append([theta,phi])
            colors.append(im[:,:,0])
            alpha = im[:,:,3]
            print('img {0} at theta {1} phi {2}'.format(f, theta, phi))
    return np.array(colors), np.array(coords), np.array(alpha)

def collect_3d_imposter_images(grid=16, filename='export/impostor_sh_N.png', frame_res=128):
    theta_step = np.pi / grid
    phi_step = np.pi * 2 / grid
    im = Image.open(filename)
    im = im.resize((frame_res*grid,frame_res*grid))
    im = np.array(im)
    w_res = frame_res
    h_res = frame_res
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
    np.save("dataset/3d_imposter", {"colors":imgs, "coords":angles})

def prepare_3d_imposter_train_dataloader(path="dataset/3d_imposter.npy", batch_size=10):
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
            num_workers=1,
        )
        return loader

    return loader(train_colors, train_coords, batch_size), loader(test_colors, test_coords, batch_size)