#!/usr/bin/env python3


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import ngff_zarr

import scipy, numpy
import sys, os, json

base_config = {
    "images" : "images.zarr",
    "masks" : "masks.zarr",
    "model" : "dt-model.pth",
    "filters" : 16,
    "depth": 3,
    "batch_size" : 128
    }

def getConfig(config_name):
    config = {}
    if os.path.exists(config_name):
        with open(sys.argv[2], 'r') as fp:
            config = json.load(fp)
    else:
        config.update(base_config)
        json.dump( config, open(sys.argv[2], 'w'), indent = 1 )
    return config

class SkipLayer(nn.Module):
    """
        Three conv3d layers, two consecutive and one that skips.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.a = nn.Sequential(
            nn.Conv3d( in_c, out_c//2, 3, padding="same" ),
            nn.Conv3d( out_c//2, out_c//2, 3, padding="same"),
            nn.ReLU() )
        self.b = nn.Sequential( nn.Conv3d( in_c, out_c//2, 3, padding="same" ), nn.ReLU() )

    def forward(self, x):
        n0 = self.a(x)
        n1 = self.b(x)
        return torch.cat([n0, n1], axis=1)

class Upscale(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.f = torch.nn.ConvTranspose3d( filters, filters, kernel_size = 2, stride = 2 )
    def forward(self, x):
        return self.f(x)

class ImageToDistanceTransform(nn.Module):
    """
        The model used for creating meshes.
    """
    def __init__(self, filters, depth):
        super().__init__()
        self.contracting = nn.ModuleList()
        self.expanding = nn.ModuleList()
        n = filters
        current = 1
        for i in range(depth):
            if( i > 0):
                self.contracting.append( nn.Sequential( nn.MaxPool3d(2), SkipLayer( current, n ) ) )
            else:
                self.contracting.append( SkipLayer( current, n ) )
            current = n
            n = 2*n

        for i in range(depth - 1):
            if( i > 0 ):
                self.expanding.append( nn.Sequential( SkipLayer( 2*current, current//2), Upscale(current//2) ) )
            else:
                self.expanding.append( nn.Sequential( SkipLayer( current, current//2 ), Upscale(current//2) ) )
            current = current // 2
        self.expanding.append( SkipLayer( 2*current, current//2 ) )
        current = current//2
        self.output = nn.Conv3d( current, 1, 3, padding="same" )

    def forward(self, x):
        skips = []
        x0 = x
        for layer in self.contracting:
            x0 = layer( x0 )
            skips.append( x0 )
        ns = len(skips)
        for i, layer in enumerate(self.expanding):
            if i > 0:
                #last layer of skips is not getting passed across.
                y0 = skips[ns - i - 1]
                x0 = torch.cat([y0, x0], axis=1)
            x0 = layer(x0)
        return self.output(x0)


class ImageDtDataset(Dataset):
    """
        The images are crops saved as zarr files. The meshes are a
    binary file.


    """
    def __init__(self, img_path, mask_path):
        img_ms = ngff_zarr.from_ngff_zarr(img_path)
        img = img_ms.images[0].data

        msk_ms = ngff_zarr.from_ngff_zarr(mask_path)
        msk = msk_ms.images[0].data
        self.masks = msk
        self.images = img


    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        x = numpy.array(self.images[idx], dtype="float32")
        y = scipy.ndimage.distance_transform_edt( numpy.array(self.masks[idx], dtype="float32") ) + numpy.array( self.masks[idx], dtype="float32" )
        #y = numpy.array( self.masks[idx], dtype="float32" )
        return x,y


def train(dataloader, model, loss_fn, optimizer, device):
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

def trainModel( config ):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    model = ImageToDistanceTransform( config["filters"], config["depth"])
    if os.path.exists( config["model"] ):
        model.load_state_dict(torch.load(config["model"], weights_only=True))

    dataset = ImageDtDataset( config["images"], config["masks"] )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle = True)
    logname = config["model"].replace(".pth", "")
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
    loss_fn = nn.MSELoss()

    for i in range(1000):
        l = train(loader, model, loss_fn, optimizer, device)
        with open("log-%s.txt"%logname, 'a') as logit:
            logit.write("%s\t%s\n"%(i, l.item() ) )
        torch.save(model.state_dict(), config["model"])

class DaskDataset(Dataset):
    def __init__(self, da):
        self.da = da
    def __len__(self):
        return self.da.shape[0]
    def __getitem__(self, idx):
        return numpy.array(self.da[idx], dtype="float32")
from matplotlib import pyplot

def predictDt( config, img_path ):
    model = ImageToDistanceTransform( config["filters"], config["depth"])
    if os.path.exists( config["model"] ):
        model.load_state_dict(torch.load(config["model"], weights_only=True))
    else:
        print("No model weights exist at %s"%config["model"])
        sys.exit(0)
    multiscales = ngff_zarr.from_ngff_zarr(img_path)
    meta = multiscales.images[0]
    img = meta.data
    bs = config["batch_size"]
    ds = DaskDataset(img)
    store = "testing.zarr"
    dl = DataLoader( ds, batch_size=bs, shuffle=False );
    out = numpy.zeros( (bs, *img.shape[1:]) )

    for x in dl:
        out[:] = model(x).detach()

    oi = ngff_zarr.to_ngff_image(out, dims=meta.dims, translation=meta.translation, scale=meta.scale)
    ms = ngff_zarr.to_multiscales( oi, scale_factors = 48 )
    print(len(ms.images), "images created")
    ngff_zarr.to_ngff_zarr( store, ms, overwrite=False )


if __name__=="__main__":
    config = getConfig(sys.argv[2])
    if sys.argv[1] == 't':
        trainModel(config)
    elif sys.argv[1] == 'p':
        predictDt(config, sys.argv[3])

