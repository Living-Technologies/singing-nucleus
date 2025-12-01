#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import ngff_zarr
import numpy
import struct
import sys, json, math, os

from singing_nucleus import SkipLayer, getConfig

import binarymeshformat as bmf

"""
This will use a NN to create a mesh output from a cropped image.
The structure of the meshes is well defined defined each mesh has a
fixed number of points.

The input will be an image crop and the output will be the positions of
the mesh.

"""

base_config = {
    "images" : "masks.zarr",
    "meshes" : "qm_mesh-masks.dat",
    "model" : "dev-model.pth",
    "latent_size" : 64,
    "filters" : 16,
    "max_pools": 3,
    "batch_size" : 128,
    "dt" : True,
    "length" : 48
    }



class BinaryToMeshTransformer(nn.Module):
    """
        The model used for creating meshes.
    """
    def __init__(self, latent_size, filters, max_pools, length):
        super().__init__()
        layers = []
        n = filters
        current = 1
        for i in range(3):
            layers.append( SkipLayer( current, n) )
            current = n
            n = 2*n
            if i < max_pools:
                layers.append( nn.MaxPool3d(2) )
        layers.append( SkipLayer(current, 2) )
        layers.append( nn.Flatten() )
        length = length//2**max_pools
        total = length**3*2
        layers.append( nn.Linear( total, latent_size ) )
        layers.append( nn.ReLU() )
        layers.append( nn.Linear( latent_size, 1926) )
        
        self.encoder = nn.Sequential(  *layers )
    def forward(self, x):
        return self.encoder(x)

def loadModel( config ):
    return BinaryToMeshTransformer( config["latent_size"], config["filters"], config["max_pools"], config["length"] )

def loadMeshFile( mesh_file, limit=-1 ):
    with open(mesh_file, 'rb') as opend:
        n = opend.read(4)
        n_indexes = struct.unpack_from(">i", n, 0)[0]
        n_bytes = n_indexes*4
        tindexes = struct.unpack_from(">%si"%n_indexes, opend.read(n_bytes), 0)

        n = opend.read(4)
        n_indexes = struct.unpack_from(">i", n, 0)[0]
        n_bytes = n_indexes*4
        cindexes = struct.unpack_from(">%si"%n_indexes, opend.read(n_bytes), 0)

        n = opend.read(4)
        values = []

        frame = 0
        while len(n) == 4:
            n_floats = struct.unpack_from(">i", n, 0)[0]
            n_bytes = n_floats*8
            f_bytes = opend.read(n_bytes)
            if len(f_bytes) != n_bytes:
                print("final mesh truncated!")
                break
            positions = struct.unpack_from(">%sd"%n_floats, f_bytes, 0)
            if any( math.isnan(x) for x in positions ):
                print(frame, "broken mesh")
            else:
                values.append(numpy.array(positions, dtype="float32"))
            n = opend.read(4)
            
            frame += 1
            if frame==limit:
                break
        return {"triangles" : tindexes,"connections":cindexes}, numpy.array(values, dtype="float32")

class ImageMeshDataset(Dataset):
    """
        The images are crops saved as zarr files. The meshes are a
    binary file.


    """
    def __init__(self, img_path, mesh_path):
        indexes, meshes = loadMeshFile(mesh_path)
        print(meshes.shape)
        multiscales = ngff_zarr.from_ngff_zarr(img_path)
        top = multiscales.images[0].data
        self.meshes = meshes;
        self.images = top[:meshes.shape[0]]
        self.topo = indexes


    def __len__(self):
        return self.meshes.shape[0]

    def __getitem__(self, idx):
        return numpy.array(self.images[idx], dtype="float32"), numpy.array(self.meshes[idx], dtype="float32")

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    acc = 0.0
    n = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        acc += loss
        n += 1
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return acc/n



def trainModel( config ):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    images = config["images"]
    meshes = config["meshes"]
    dataset = ImageMeshDataset(images, meshes)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle = True)
    transformer = loadModel(config)
    
    if os.path.exists( config["model"] ):
        transformer.load_state_dict(torch.load(config["model"], weights_only=True))

    model  = transformer.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), lr = 0.001)
    logname = config["model"].replace(".pth", "")
    for i in range(1000):
        l = train(loader, model, loss_fn, optimizer, device)
        with open("log-%s.txt"%logname, 'a') as logit:
            logit.write("%s\t%s\n"%(i, l.item() ) )
        torch.save(model.state_dict(), config["model"])

def predictMeshes( config, image ):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    transformer = loadModel(config)
    transformer.load_state_dict(torch.load(config["model"], weights_only=True, map_location=device))
    images = config["images"]
    meshes = config["meshes"]
    dataset = ImageMeshDataset(images, meshes)
    topo = dataset.topo
    track = bmf.Track("predicted")
    multiscales = ngff_zarr.from_ngff_zarr(image)
    top = multiscales.images[0].data
    
    dex = 0
    for i, x in enumerate(top):
        z = transformer(torch.clamp( torch.tensor(numpy.expand_dims( numpy.array(x, dtype="float32"), 0)), 0, 1) )
        for row in z:
            mesh = bmf.Mesh(row.detach(), topo["connections"], topo["triangles"])
            track.addMesh( dex, mesh)
            dex += 1
    bmf_name = "dev-%s.bmf"%config["model"].replace(".pth", "")
    bmf.saveMeshTracks( [track], bmf_name)

def saveMeshes(config):
    n = 64;
    topo, positions = loadMeshFile(config["meshes"], limit=n)
    track = bmf.Track("truth")
    for i in range(n):
        mesh = bmf.Mesh(positions[i], topo["connections"], topo["triangles"])
        track.addMesh( i, mesh)
    bmf.saveMeshTracks([track], "dev-truth.bmf")

if __name__=="__main__":
    config = getConfig(sys.argv[2], base_config)
    if sys.argv[1] == 't':
        trainModel(config)
    elif sys.argv[1] == 'p':
        predictMeshes(config, sys.argv[3])
    elif sys.argv[1]=='e':
        saveMeshes(config)
