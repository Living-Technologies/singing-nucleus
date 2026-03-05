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
from dask.distributed import Lock, Client

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
    it = skimage.measure.inertia_tensor( msk[0] )
    eig = numpy.linalg.eig( it )

    values = [ (e, *v) for e, v in zip(eig.eigenvalues, eig.eigenvectors) ]
    values.sort()
    evs = numpy.array([ev[1:] for  ev in values], dtype="float32" )
    evs = evs.reshape((9, ))
    return ( numpy.array(scipy.ndimage.distance_transform_edt( msk ), dtype="float32"), evs )

class ImageDtDataset(Dataset):
    """
        The images are crops saved as zarr files. The meshes are a
    binary file.


    """
    def __init__(self, img_paths, mask_paths):
        print(len(img_paths), "images", len(mask_paths), "masks")
        self.masks = []
        self.images = []
        self.lengths = []
        for img_path, mask_path in zip(img_paths, mask_paths):
            img_ms = ngff_zarr.from_ngff_zarr(img_path)
            img = img_ms.images[0].data

            msk_ms = ngff_zarr.from_ngff_zarr(mask_path)
            msk = msk_ms.images[0].data
            self.masks.append(msk)
            self.images.append(img)
            self.lengths.append(img.shape[0])

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        sdex = 0
        dex = idx
        while dex >= self.lengths[sdex]:
            dex = dex - self.lengths[sdex]
            sdex += 1


        x = numpy.array(self.images[sdex][dex], dtype="float32")
        y, z = dt( self.masks[sdex][dex] )
        return x, y, z


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    acc = [0.0, 0.0]
    n = 0
    for batch, (X, y, z) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        z = z.to(device)

        # Compute prediction error
        pred, mat = model(X)
        dt_loss, mat_loss = loss_fn(pred, y, mat, z)

        loss = dt_loss + mat_loss * 0.00001
        acc[0] += dt_loss.item()
        acc[1] += mat_loss.item()
        n += 1

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return [ a/n for a in acc]

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

def trainModel( config ):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    torch.multiprocessing.set_sharing_strategy('file_system')
    model = ImageToDistanceTransform( config["filters"], config["depth"])
    if os.path.exists( config["model"] ):
        model.load_state_dict(torch.load(config["model"], weights_only=True), strict=False)



    dataset = DaskingDataset( config["images"], config["masks"] )
    if "epoch_limit" in config:
        limit = config["epoch_limit"]
        print("epoch limit set to:", limit)
        dataset.setLimit(limit)

    loader = DataLoader(dataset, batch_size=config["batch_size"],num_workers = 4, prefetch_factor=8)
    logname = config["model"].replace(".pth", "")
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
    #loss_fn = nn.MSELoss()
    loss_fn = clipped_mse
    #loss_fn = PairedLossFunction()
    model.to(device)
    model.train()

    #client = Client()


    for i in range(1000):
        start = time.time();
        l = train(loader, model, loss_fn, optimizer, device)
        #l = daskTrainingLoop(dataset, model, loss_fn, optimizer, device)
        print("trained: ", (time.time() - start))
        with open("log-%s.txt"%logname, 'a') as logit:
            logit.write("%s\t%s\t%s\n"%(i, l[0], l[1] ) )
        start = time.time();
        torch.save(model.state_dict(), config["model"])
        print("saved: ", (time.time() - start))

class CustomLoader():
    def __init__(self, dataset, device, batch_size=1):
        self.dataset = dataset
        self.n = len(dataset)//batch_size
        self.batch_size = batch_size
        self.device = device
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        batchx = []
        batchy = []
        batchz = []
        low = idx*self.batch_size

        for i in range(self.batch_size):
            x, y, z = self.dataset[idx + i]
            batchx.append(x)
            batchy.append(y)
            batchz.append(z)
        device = self.device
        return torch.Tensor(numpy.array(batchx)).to(device), torch.Tensor(numpy.array(batchy)).to(device), torch.Tensor(numpy.array(batchz)).to(device)
    def generator(self):
        for i in range(self.n):
            yield self[i]

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
        self.images = da.concatenate(self.images, axis=0)[0:500]
        self.masks = da.concatenate(self.masks, axis=0)[0:500]
        self.n = self.images.shape[0]
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        x = numpy.array(self.images[idx], dtype="float32")
        y, z = getDt( self.masks[idx] )
        return x, y, z
    def setLimit(self, limit):
        self.n = limit


def daskTrainingLoop(dataset, model, loss_fn, optimizer, device):
    n = len(dataset)
    batch_size = 16

    total_batches = n//batch_size

    last = 0
    if n%batch_size > 0:
        last += 1

    batches = n//batch_size + last
    lock = Lock()

    def torchit( block_id, model=model, dataset = dataset, bs = batch_size, device=device, lock = lock ):
        idx = block_id[0]
        low = idx*bs
        high = low + bs
        elements = bs
        if high > dataset.images.shape[0]:
            high = dataset.images.shape[0]
            elements = high - low

        x = torch.tensor( numpy.array(dataset.images[low:high], dtype="float32") , device=device )
        t = [ getDt(dataset.masks[low + i]) for i in range(elements) ]
        dt = torch.tensor( numpy.array([i[0] for i in t ]), device = device )
        matrix = torch.tensor( numpy.array([i[1] for i in t]) , device = device )


        y, z = model(x)

        dt_loss, mat_loss = loss_fn(y, dt, z, matrix)

        loss = dt_loss + mat_loss * 0.00001
        lock.acquire()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        ret = numpy.array([ [loss.cpu().detach().numpy(), dt_loss.cpu().detach().numpy(), mat_loss.cpu().detach().numpy()]] )
        lock.release()
        return ret

    chunks = ( (1, ) * batches, 3 )

    out = dask.array.map_blocks( torchit, dtype="float32", chunks = chunks )
    total = out.mean(axis=0)
    return total.compute()

def daskEvaluateModel(config):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    #device = "cpu"
    print(f"Using {device} device")

    model = ImageToDistanceTransform( config["filters"], config["depth"])
    if os.path.exists( config["model"] ):
        model.load_state_dict(torch.load(config["model"], weights_only=True), strict=False)
    else:
        print("cannot find existing model: ", config["model"])
    #dataset = ImageDtDataset( config["images"], config["masks"] )
    dataset = DaskingDataset( config["images"], config["masks"] )
    total = len(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle = False, num_workers = 4, prefetch_factor = 8)
    #loader = CustomLoader(dataset, device).generator()

    logname = config["model"].replace(".pth", "")
    loss_fn = PairedLossFunction()
    model.to(device)
    model.eval()
    load_time = 0
    proc_time = 0
    loss_time = 0

    time_start = time.time()

    img = dataset.images

    bs = 32
    n = img.shape[0]
    last = n%bs
    head = (1, )*(n//bs)

    def torchit( block_id, model=model, dataset = dataset ):
        idx = block_id[0]
        low = idx*bs
        high = low + bs
        elements = bs
        if high > dataset.images.shape[0]:
            high = dataset.images.shape[0]
            elements = high - low

        x = torch.tensor( numpy.array(dataset.images[low:high], dtype="float32") , device=device )

        y, z = model(x)

        y = y.detach().cpu()
        z = z.detach().cpu()

        t = [ getDt(dataset.masks[low + i]) for i in range(elements) ]
        dt = torch.tensor( numpy.array([i[0] for i in t ]) )
        matrix = torch.tensor( numpy.array([i[1] for i in t]) )

        l0, l1 = loss_fn(y, dt, z, matrix)
        ret = numpy.array([ [l0.numpy(), l1.numpy()]] )
        return ret

    sample = img[0]
    print("preparing")

    if last == 0:
        chunks = ( (*head, ), 2 )
    else:
        chunks = ((*head, 1), 2)

    out = dask.array.map_blocks( torchit, dtype="float32", chunks = chunks )
    #out.compute_chunk_sizes()
    print("processing")
    tstart = time.time()
    out = out.compute(num_workers = 3)
    for err in out:
        start = time.time()
        val = err
        print( (time.time() - start), "batch complete" )
    print((time.time() - tstart), "completed")

def evaluateModel(config):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    model = ImageToDistanceTransform( config["filters"], config["depth"])
    if os.path.exists( config["model"] ):
        model.load_state_dict(torch.load(config["model"], weights_only=True), strict=False)
    else:
        print("cannot find existing model: ", config["model"])
    #dataset = ImageDtDataset( config["images"], config["masks"] )
    dataset = DaskingDataset( config["images"], config["masks"] )
    total = len(dataset)
    loader = DataLoader(dataset, batch_size=1, shuffle = False, num_workers = 4, prefetch_factor = 8)

    logname = config["model"].replace(".pth", "")
    loss_fn = PairedLossFunction()
    model.to(device)
    model.eval()
    load_time = 0
    proc_time = 0
    loss_time = 0

    time_start = time.time()
    first_start = time.time()
    with open("error-%s.txt"%logname, 'w') as recording:
        for batch, (X, y, z) in enumerate(loader):
            if batch%100 == 0:
                print(batch, "of", total, " load: ", load_time, "pred:", proc_time, "loss:", loss_time)
                recording.flush()
                load_time = 0
                proc_time = 0
                loss_time = 0
            vol = torch.sum( ( y > 0 ) * 1.0 ).item()
            X = X.to(device)

            load_time += time.time() - time_start

            time_start = time.time()
            pred, matrix = model(X)
            proc_time += time.time() - time_start
            time_start = time.time()
            pred = pred.cpu()
            matrix = matrix.cpu()
            loss, loss2 = loss_fn(pred, y, matrix, z)
            loss_time += time.time() - time_start

            time_start = time.time()
            recording.write("%s\t%s\t%s\t%s\n"%(batch, loss.item(), loss2.item(), vol))
            # Backpropagation
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
        evaluateModel(config)
        #daskEvaluateModel(config)

