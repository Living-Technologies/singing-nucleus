#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import ngff_zarr
import numpy
import struct
import sys, json, math, os, pathlib



from bin_2_mesh import BinaryToMeshTransformer, ImageMeshDataset
from img_2_dt import ImageToDistanceTransform
from singing_nucleus import getConfig

base_config = {
    "images" : "images.zarr",
    "meshes" : "qm_mesh-masks.dat",
    "model" : "full-model.pth",
    "dt_model" : {
            "model" : "dt-model.pth",
            "filters" : 16,
            "depth": 3
        },
    "mesh_model" : {
            "model" : "dev-model.pth",
            "latent_size" : 64,
            "filters" : 16,
            "max_pools": 3
        },
    "batch_size" : 128
    }

class ImageToMeshTransformer(nn.Module):
    def __init__(self, imgToDt, binToMesh):
        super().__init__()
        self.imgToDt = imgToDt
        self.binToMesh = binToMesh
    def forward( self, x):
        return self.binToMesh( self.imgToDt( x ) )

def trainStep(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.to(device)
    model.train()
    acc = 0.0
    n = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        acc += loss
        n += 1
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return acc/n

def train(config):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    dataset = ImageMeshDataset(config["images"], config["meshes"])
    unet = config["dt_model"]
    img2Dt = ImageToDistanceTransform( unet["filters"], unet["depth"] )
    mconf = config["mesh_model"]
    bin2Mesh = BinaryToMeshTransformer( mconf["latent_size"], mconf["filters"], mconf["max_pools"] )
    model = ImageToMeshTransformer( img2Dt, bin2Mesh )
    mdlPath = pathlib.Path(config["model"])
    if mdlPath.exists():
        model.load_state_dict(torch.load(config["model"], weights_only=True, map_location=device))
    else:
        model.imgToDt.load_state_dict(torch.load(unet["model"], weights_only=True, map_location=device))
        model.binToMesh.load_state_dict(torch.load(mconf["model"], weights_only=True, map_location=device))
    print("loaded and training!")

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle = True)
    logname = config["model"].replace(".pth", "")
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
    loss_fn = nn.MSELoss()

    for i in range(1000):
        l = trainStep(loader, model, loss_fn, optimizer, device)
        with open("log-%s.txt"%logname, 'a') as logit:
            logit.write("%s\t%s\n"%(i, l.item() ) )
        torch.save(model.state_dict(), config["model"])

if __name__ == "__main__":
    print("running")
    config = getConfig( sys.argv[2], base_config )
    if sys.argv[1] == 't':
        train(config)
