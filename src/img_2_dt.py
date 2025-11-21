#!/usr/bin/env python3


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import ngff_zarr

import scipy, numpy
import sys, os, json

import dask

from singing_nucleus import getConfig, SkipLayer

base_config = {
    "images" : "images.zarr",
    "masks" : "masks.zarr",
    "model" : "dt-model.pth",
    "filters" : 16,
    "depth": 3,
    "batch_size" : 128
    }


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
        print(img_path, mask_path)
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
        y = numpy.array( scipy.ndimage.distance_transform_edt( self.masks[idx] ), dtype="float32")
        return x,y


def train(dataloader, model, loss_fn, optimizer, device, limit=100):
    size = len(dataloader.dataset)
    model.to(device)
    model.train()
    acc = 0.0
    n = 0

    if not limit > 0:
        limit = size
    count = 0
    for batch, (X, y) in enumerate(dataloader):
        if count >= limit:
            break
        count +=  1
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
    limit = None
    if "epoch_limit" in config:
        limit = config["epoch_limit"]
    
    print("epoch limit set to:", limit)
    for i in range(1000):
        l = train(loader, model, loss_fn, optimizer, device, limit = limit)
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

def predictDt( config, img_path ):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = ImageToDistanceTransform( config["filters"], config["depth"])
    if os.path.exists( config["model"] ):
        model.load_state_dict(torch.load(config["model"], weights_only=True, map_location=device))
    else:
        print("No model weights exist at %s"%config["model"])
        sys.exit(0)
    model.to(device)

    multiscales = ngff_zarr.from_ngff_zarr(img_path)
    meta = multiscales.images[0]
    img = meta.data

    bs = 4
    n = img.shape[0]
    last = n%bs
    head = (bs, )*(n//bs)
    store = "testing.zarr"

    stacks = []
    def torchit( block_id, model=model, img=img, bs=bs ):
        idx = block_id[0]
        low = idx*bs
        high = low + bs
        x = torch.tensor( numpy.array(img[low:high], dtype="float32") , device=device )
        y = model(x).clip(0, 127).detach().to( torch.int8).cpu()
        return y

    sample = img[0]
    print("preparing")
    out = dask.array.map_blocks( torchit, dtype="int8", chunks = ((*head, last), *sample.shape) )
    print( "out stack: ", out.shape )
    oi = ngff_zarr.to_ngff_image(out, dims=meta.dims, translation=meta.translation, scale=meta.scale)
    ms = ngff_zarr.to_multiscales( oi, cache=False, chunks=(1, 1, 48, 48, 48) )
    print("writing")
    ngff_zarr.to_ngff_zarr( store, ms, overwrite=False )

def validateConfig( config ):
    from matplotlib import pyplot
    dataset = ImageDtDataset( config["images"], config["masks"] )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle = False)

    model = ImageToDistanceTransform( config["filters"], config["depth"])
    if os.path.exists( config["model"] ):
        model.load_state_dict(torch.load(config["model"], weights_only=True))


    for x, y in loader:
        line = y[0, 0, 24, 24].detach()
        pyplot.plot(line)
        z = model(x)
        line2 = z[0, 0, 24, 24].detach()
        pyplot.plot(line2)
        pyplot.show()
        print("max loaded vs predicted", torch.max(y).item(), torch.max(z).item() )

def createModelConfig(config, config_path, img_path, mask_path):
    print(config_path)
    config["images"] = img_path
    config["masks"] = mask_path
    if os.path.exists(img_path):
        print("images found: ", img_path)
    if os.path.exists(mask_path):
        print("masks found: ", mask_path)
    config["model"] = config_path.replace(".json", ".pth")
    with open(config_path, 'w') as f:
        json.dump( config, f, indent = 1 )

        
if __name__=="__main__":
    commands =  ['t', 'p', 'v', 'c'] 
    if not sys.argv[1] in commands:
        print("Choose a command!", commands)
        sys.exit(1)
    config = getConfig(sys.argv[2], base_config)
    if sys.argv[1] == 't':
        trainModel(config)
    elif sys.argv[1] == 'p':
        predictDt(config, sys.argv[3])
    elif sys.argv[1] == 'v':
        validateConfig( config )
    elif sys.argv[1] =='c':
        try:
            createModelConfig(config, sys.argv[2], sys.argv[3], sys.argv[4])
            print("successfuly saved config", sys.argv[2])            
        except:
            print("usage: img_2_dt c config.json image_path mask_path")
            for item in sys.exc_info():
                print(item)


