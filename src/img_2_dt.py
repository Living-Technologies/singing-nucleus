#!/usr/bin/env python3


import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import ngff_zarr

import scipy, numpy
import sys, os, json

import dask

import time
import skimage

import dask.array as da
import concurrent.futures

import random

from singing_nucleus import getConfig, SkipLayer

base_config = {
    "images" : ["images.zarr"],
    "masks" : ["masks.zarr"],
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
    def __init__(self, filters, depth, crop_size=64):
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
        # starting edge 64
        l = crop_size//(2**(depth-1))
        self.matrix = nn.Sequential( torch.nn.Flatten(), nn.Linear(l*l*l*current, 9), torch.nn.Sigmoid() )
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
        matrix = self.matrix( x0 )
        for i, layer in enumerate(self.expanding):
            if i > 0:
                #last layer of skips is not getting passed across.
                y0 = skips[ns - i - 1]
                x0 = torch.cat([y0, x0], axis=1)
            x0 = layer(x0)
        return self.output(x0), matrix

def getDt(img):
    msk = numpy.array(img, dtype="float32")
    try:
        it = skimage.measure.inertia_tensor( msk[0] )
        eig = numpy.linalg.eig( it )
        values = [ (e, *v) for e, v in zip(eig.eigenvalues, eig.eigenvectors) ]
        values.sort()
        evs = numpy.array([ev[1:] for  ev in values], dtype="float32" )
        evs = evs.reshape((9, ))
    except:
        evs = numpy.zeros((9, ))
    return ( numpy.array(scipy.ndimage.distance_transform_edt( msk ), dtype="float32"), evs )


class QueueLoader:
    def __init__(self, dataset, batch_size, device, parallel = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.n = len(self.dataset)
        if self.n % batch_size == 0:
            self.batches = self.n//self.batch_size
        else:
            self.batches = self.n//self.batch_size + 1
        self.parallel = parallel

    def runMe(self, b):
        idx = b
        low = idx * self.batch_size
        high = low + self.batch_size
        elements = self.batch_size

        if high > self.n:
            high = self.n
            elements = high - low

        images = self.dataset[low:high]
        x = torch.tensor( images[0], device = self.device )
        dt = torch.tensor( images[1], device = self.device )
        matrix = torch.tensor( images[2], device = self.device )

        return (x, dt, matrix)

    def __iter__(self):
        ex = concurrent.futures.ThreadPoolExecutor()

        jobs = [(self.runMe, i) for i in range(self.batches)]
        L = []
        for i in range(self.parallel):
            job = jobs.pop(0)
            L.append( ex.submit(*job) )
            if len(jobs)==0:
                break
        out = numpy.zeros((3, ))
        while len(L) > 0:
            future = L.pop(0)
            ret = future.result()
            if len(jobs) > 0:
                job = jobs.pop(0)
                L.append(ex.submit(*job))

            yield ret
        ex.shutdown()

class DaskingDataset():
    """
        The images are crops saved as zarr files. The meshes are a
    binary file.


    """
    def __init__(self, img_paths, mask_paths):
        print(len(img_paths), "images", len(mask_paths), "masks")
        self.masks = []
        self.images = []
        for img_path, mask_path in zip(img_paths, mask_paths):
            img_ms = ngff_zarr.from_ngff_zarr(img_path)
            img = img_ms.images[0].data

            msk_ms = ngff_zarr.from_ngff_zarr(mask_path)
            msk = msk_ms.images[0].data
            self.masks.append(msk)
            self.images.append(img)
        self.images = da.concatenate(self.images, axis=0)
        self.masks = da.concatenate(self.masks, axis=0)
        self.n = self.images.shape[0]
        self.indexes = [i for i in range(self.n)]
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        idx = self.indexes[i]
        if not isinstance(idx, list):
            x = numpy.array(self.images[idx, ...], dtype="float32")
            mask_set = self.masks[idx, ...]
            y, z = getDt(mask_set)
        else:
            x = numpy.array(self.images[idx, ...], dtype="float32")
            mask_set = self.masks[idx, ...]
            dt_mat = [getDt(mask) for mask in mask_set ]
            y = numpy.array([row[0] for row in dt_mat ])
            z = numpy.array([row[1] for row in dt_mat ])
        return x, y, z

    def shuffle(self):
        random.shuffle(self.indexes)
    def setLimit(self, limit):
        if limit > len(self.indexes):
            print("warning: limit is not set, too large for dataset.", limit, " is more than ", self.n)
        else:
            self.n = limit

def epoch(dataset, model, loss_fn, optimizer):
    out = numpy.zeros((3, ), dtype=float)
    batches = 0
    for x, dt, matrix in dataset:
        y, z = model(x)
        dt_loss, mat_loss = loss_fn(y, dt, z, matrix)
        loss = dt_loss + mat_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        out[0] += loss.item()
        out[1] += dt_loss.item()
        out[2] += mat_loss.item()
        batches += 1

    return out/batches

def clipped_mse(pred, truth, pred_mat, mat):
    vol = torch.sum( (truth > 0 )*1.0, (-3, -2, -1), keepdims=True)
    clip = ( vol>7 ) * 1.0

    #number of elements calculating the error for
    n = truth.shape[-3]*truth.shape[-2]*truth.shape[-1] * torch.sum(clip)
    if n == 0:
        n = 1

    delta = (pred - truth)*clip
    mse = torch.sum(delta*delta)/n
    clip2 = torch.squeeze(clip, (-3, -2, -1))
    n2 = mat.shape[-1]*torch.sum(clip2)
    delta2 = (pred_mat - mat)*clip2

    #number of vector elements.
    if n2 == 0:
        n2 = 1

    mse2 = torch.sum(delta2*delta2)/n2
    return mse, mse2

class PairedLossFunction:
    def __init__(self):
        self.mse = nn.MSELoss()
    def __call__(self, pred, truth, pred_mat, mat):
        return self.mse(pred, truth), self.mse(pred_mat, mat)

class DeviceLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
    def __iter__(self, ):
        for item in self.loader:
            yield tuple( i.to(self.device) for i in item )

def trainModel( config ):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = ImageToDistanceTransform( config["filters"], config["depth"])
    if os.path.exists( config["model"] ):
        model.load_state_dict(torch.load(config["model"], weights_only=True, map_location=device), strict=False)



    dataset = DaskingDataset( config["images"], config["masks"] )

    if "epoch_limit" in config:
        limit = config["epoch_limit"]
        print("epoch limit set to:", limit)
        dataset.setLimit(limit)

    model.to(device)
    model.train()
    loss_fn = clipped_mse
    logname = config["model"].replace(".pth", "")
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)

    use_torch_dataload = "PYTORCH_DATALOADER" in os.environ
    if use_torch_dataload:
        print("using pytorch dataloader")
        pin_memory = device != "mps"
        loader = DataLoader(dataset, batch_size=config["batch_size"],num_workers = 4, prefetch_factor=8, pin_memory=pin_memory)
        loader = DeviceLoader(loader, device)
    else:
        loader = QueueLoader(dataset, config["batch_size"], device)
    for i in range(1000):
        start = time.time();
        dataset.shuffle()
        l = epoch(loader, model, loss_fn, optimizer)
        print("trained: ", (time.time() - start))
        with open("log-%s.txt"%logname, 'a') as logit:
            logit.write("%s\t%s\t%s\t%s\n"%(i, l[0], l[1], l[2] ) )
        start = time.time();
        torch.save(model.state_dict(), config["model"])
        print("saved: ", (time.time() - start))


def evaluateModel(config):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = ImageToDistanceTransform( config["filters"], config["depth"])
    if os.path.exists( config["model"] ):
        model.load_state_dict(torch.load(config["model"], weights_only=True, map_location=device), strict=False)
    else:
        print("cannot find existing model: ", config["model"])


    batch_size = 1

    logname = config["model"].replace(".pth", "")
    loss_fn = PairedLossFunction()
    model.to(device)
    model.eval()
    load_time = 0
    proc_time = 0
    loss_time = 0

    time_start = time.time()
    first_start = time.time()
    use_torch_dataloader = False

    average_values = []
    with open("error-%s.txt"%logname, 'w') as recording:

        samples = 0
        for pair in zip(config["images"], config["masks"]):
            print("pair", pair)
            dataset = DaskingDataset([pair[0]], [pair[1]])
            if use_torch_dataloader:

                loader = DataLoader( dataset, batch_size=1, shuffle = False, num_workers = 4, prefetch_factor = 8)
                loader = DeviceLoader(loader, device)
            else:
                loader = QueueLoader(dataset, 1, device, parallel=6)
            cum_loss = numpy.zeros((2, ))
            cum_loss_2 = numpy.zeros((2, ))
            n = 0
            for batch,data in enumerate(loader):
                X, y, z = data
                vol = torch.sum( 1.0*(y>0) )
                if vol < 10:
                    continue
                samples += 1
                if samples%100 == 0:
                    print(samples, " load: ", load_time, "pred:", proc_time, "loss:", loss_time)
                    recording.flush()
                    load_time = 0
                    proc_time = 0
                    loss_time = 0
                    break

                load_time += time.time() - time_start

                time_start = time.time()
                pred, matrix = model(X)
                proc_time += time.time() - time_start
                time_start = time.time()
                loss, loss2 = loss_fn(pred, y, matrix, z)
                loss_time += time.time() - time_start
                cum_loss[0] += loss.item()
                cum_loss[1] += loss2.item()
                cum_loss_2[0] += loss.item()*loss.item()
                cum_loss_2[1] += loss2.item()*loss2.item()
                time_start = time.time()
                recording.write("%s\t%s\t%s\n"%(batch, loss.item(), loss2.item()))
                n += 1
            cum_loss = cum_loss/n
            var = cum_loss_2/n - cum_loss*cum_loss
            average_values.append( ( pair[0], cum_loss[0], var[0], cum_loss[1], var[1], n ) )
    with open("ave_%s.txt"%logname, 'w') as lo:
        for line in average_values:
            sline = "\t".join( "%s"%item for item in line )
            lo.write("%s\n"%sline)
    print("total: ", (time.time() - first_start))

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
        y = model(x)[0].clip(0, 127).detach().to( torch.int8).cpu()
        return y

    sample = img[0]
    print("preparing")
    out = dask.array.map_blocks( torchit, dtype="int8", chunks = ((*head, last), *sample.shape) )
    print( "out stack: ", out.shape )
    oi = ngff_zarr.to_ngff_image(out, dims=meta.dims, translation=meta.translation, scale=meta.scale)
    ms = ngff_zarr.to_multiscales( oi, cache=False, chunks=(1, 1, *out.shape[2:]) )
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
    config["images"] = [img_path]
    config["masks"] = [mask_path]
    if os.path.exists(img_path):
        print("images found: ", img_path)
    if os.path.exists(mask_path):
        print("masks found: ", mask_path)
    config["model"] = config_path.replace(".json", ".pth")
    with open(config_path, 'w') as f:
        json.dump( config, f, indent = 1 )

        
if __name__=="__main__":
    commands =  ['t', 'p', 'v', 'c', 'e']
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
    elif sys.argv[1] == 'e':
        with torch.no_grad():
            evaluateModel(config)
            #daskEvaluateModel(config)

