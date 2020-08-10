# PyTorch Distributed K-FAC Preconditioner

[![DOI](https://zenodo.org/badge/240976400.svg)](https://zenodo.org/badge/latestdoi/240976400)

Code for the paper "[Convolutional Neural Network Training with Distributed K-FAC](https://arxiv.org/abs/2007.00784)."

The KFAC code was originally forked from Chaoqi Wang's [KFAC-PyTorch](https://github.com/alecwangcq/KFAC-Pytorch).
The ResNet models for Cifar10 are from Yerlan Idelbayev's [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10).
The CIFAR-10 and ImageNet-1k training scripts are modeled afer Horovod's example PyTorch training scripts.

## Install

### Requirements

KFAC supports [Horovod](https://github.com/horovod/horovod) and `torch.distributed` distributed training backends.

This code is validated to run with PyTorch >=1.2.0, Horovod >=0.19.0, and CUDA >=10.0.

### Installation

```
$ git clone https://github.com/gpauloski/kfac_pytorch.git
$ cd kfac_pytorch
$ pip install .
```

## Usage

The K-FAC Preconditioner can be easily added to exisiting training scripts that use `horovod.DistributedOptimizer()`.

```Python
from kfac import KFAC
... 
optimizer = optim.SGD(model.parameters(), ...)
optimizer = hvd.DistributedOptimizer(optimizer, ...)
preconditioner = KFAC(model, ...)
... 
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.synchronize()
    preconditioner.step()
    with optimizer.skip_synchronize():
        optimizer.step()
...
```

Note that the K-FAC Preconditioner expects gradients to be averaged across workers before calling `preconditioner.step()` so we call `optimizer.synchronize()` before hand (Normally `optimizer.synchronize()` is not called until `optimizer.step()`). 

For `torch.distributed` or non-distributed scripts, just call `KFAC.step()` before `optimizer.step()`. KFAC will automatically determine if training is being done in a distributed way and what backend is being used.

## Example Scripts

Example scripts for K-FAC + SGD training on CIFAR-10 and ImageNet-1k are provided.
For a full list of training parameters, use `--help`, e.g. `python examples/horovod_cifar10_resnet.py --help`.
Package requirements for the examples are given in [examples/README.md](examples/README.md).

### Horovod

```
$ mpiexec -hostfile /path/to/hostfile -N $NGPU_PER_NODE python examples/horovod_{cifar10,imagenet}_resnet.py
```

### torch.distributed

#### Single Node, Multi-GPU
```
$ python -m torch.distributed.launch --nproc_per_node=$NGPU_PER_NODE examples/torch_{cifar10,imagenet}_resnet.py
```

#### Multi-Node, Multi-GPU
On each node, run:
```
$ python -m torch.distributed.launch \
          --nproc_per_node=$NGPU_PER_NODE --nnodes=$NODE_COUNT \
          --node_rank=$NODE_RANK --master_addr=$MASTER_HOSTNAME \
      examples/torch_{cifar10,imagenet}_resnet.py
```
Note: if using model parallel training as well (i.e. the model is split across multiple GPUs), KFAC will perform the ops for each module on the device the module is on. So if a model is split across two GPUs, the KFAC factors and inverses will also be split across GPUs.

### Mixed-Precision Training
KFAC will not work with NVIDIA AMP training because the inverse operations used in KFAC (`torch.inverse` and `torch.symeig`) do not support half-precision inputs, and NVIDIA AMP does allow for disabling autocast in certain code regions.
The `experimental` KFAC branch has worked with `torch.cuda.amp` and `torch.nn.parallel.DistributedDataParallel` although this support is still considered experimental.
Note that this will require PyTorch 1.6 or newer.
When using `torch.cuda.amp` for mixed precision training, be sure to call `KFAC.step()` outside of an `autocast()` region. E.g.
```Python
...
scaler = GradScaler()
...
for i, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # Unscale gradients before KFAC.step()
    preconditioner.step()
    scaler.step(optimizer)
    scaler.update()
...
```
For more help with gradient accumulation or model-parallel training, see the [PyTorch mixed-precision docs](https://pytorch.org/docs/stable/notes/amp_examples.html#typical-mixed-precision-training).

## Citation

```
@article{pauloski2020convolutional,
    title={Convolutional Neural Network Training with Distributed K-FAC},
    author={J. Gregory Pauloski and Zhao Zhang and Lei Huang and Weijia Xu and Ian T. Foster},
    year={2020},
    pages={to appear in the proceedings of SC20},
    eprint={2007.00784},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
