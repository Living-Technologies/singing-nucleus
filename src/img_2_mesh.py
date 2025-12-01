#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import ngff_zarr
import numpy
import struct
import sys, json, math, os, pathlib
import binarymeshformat as bmf


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
            "max_pools": 3,
            "length" : 48
        },
    "batch_size" : 16
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
def loadModel( config, device=None ):
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
    return model
    
def train(config):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    dataset = ImageMeshDataset(config["images"], config["meshes"])
    model = loadModel( config )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle = True)
    logname = config["model"].replace(".pth", "")
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
    loss_fn = nn.MSELoss()

    for i in range(1000):
        l = trainStep(loader, model, loss_fn, optimizer, device)
        with open("log-%s.txt"%logname, 'a') as logit:
            logit.write("%s\t%s\n"%(i, l.item() ) )
        torch.save(model.state_dict(), config["model"])

def createModel( config, config_path, dt_config, mesh_config):
    
    with open(dt_config, 'r') as fp:
            dtcfg = json.load(fp)
            to_update = config["dt_model"]
            for key in to_update:
                to_update[key] = dtcfg[key]
    with open(mesh_config, 'r') as fp:
            meshcfg = json.load(fp)
            to_update = config["mesh_model"]
            for key in to_update:
                to_update[key] = meshcfg[key]
    config["images"] = dtcfg["images"]
    config["meshes"] = meshcfg["meshes"]
    json.dump( config, open(config_path, 'w'), indent = 1 )

def loadTopology( quickmesh_path ):
    with open(quickmesh_path, 'rb') as opend:
        n = opend.read(4)
        n_indexes = struct.unpack_from(">i", n, 0)[0]
        n_bytes = n_indexes*4
        tindexes = struct.unpack_from(">%si"%n_indexes, opend.read(n_bytes), 0)

        n = opend.read(4)
        n_indexes = struct.unpack_from(">i", n, 0)[0]
        n_bytes = n_indexes*4
        cindexes = struct.unpack_from(">%si"%n_indexes, opend.read(n_bytes), 0)
    return {"triangles" : tindexes,"connections":cindexes}
    
def predict( config, config_path, image_path, output=None):
    if output is None:
        output = image_path.replace(".zarr", "-pred.zarr")
        
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    transformer = loadModel(config, device)

    images = config["images"]
    meshes = config["meshes"]

    #dataset = ImageMeshDataset(images, meshes)
    topo = loadTopology( meshes )
    track = bmf.Track("predicted")
    multiscales = ngff_zarr.from_ngff_zarr(image_path)
    top = multiscales.images[0].data
    
    dex = 0
    for i, x in enumerate(top):
        z = transformer(torch.tensor(numpy.expand_dims( numpy.array(x, dtype="float32"), 0)))
        for row in z:
            mesh = bmf.Mesh(row.detach(), topo["connections"], topo["triangles"])
            track.addMesh( dex, mesh)
            dex += 1
    bmf_name = "dev-%s.bmf"%config["model"].replace(".pth", "")
    bmf.saveMeshTracks( [track], bmf_name)
    
if __name__ == "__main__":
    print("running")
    config = getConfig( sys.argv[2], base_config )
    
    if sys.argv[1] == 't':
        train(config)
    elif sys.argv[1] == 'c':
        try:
            createModel( config, sys.argv[2], sys.argv[3], sys.argv[4] )    
        except:
            print("usage: img_2_mesh c config.json dt_config.json mesh_config.json")
            print(sys.exc_info())
    elif sys.argv[1] == 'p':
        try:
            predict( config, sys.argv[2], sys.argv[3], output=None)
        except:
            print("usage: img_2_mesh p config.json sample.zarr")
            print(sys.exc_info())
