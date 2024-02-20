# Compressing Latent Space via Least Volume (ICLR 2024)

This is the official GitHub repository for the ICLR 2024 paper [**_Compressing Latent Space via Least Volume_**](https://openreview.net/forum?id=jFJPd9kIiF&noteId=jFJPd9kIiF).


## Environment
Install Anaconda environment `lv` with the following command:
```
conda create lv python=3.10 jupyter matplotlib scikit-learn pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Key Components

Unless you are interested in verifying the results in this paper, there is no need to tramp through this repo to apply Least Volume to your own project. As introduced in §2, there are only two key components to extract from this work for your own use:

### 1. Volume Penalty
The volume penalty is defined on Line [#37](./src/model/sparsity.py#L37) of [sparsity.py](./src/model/sparsity.py) as:
```
torch.exp(torch.log(z.std(0) + η).mean())
```
where `z` is the latent code. Just append this penalty to your exisiting loss function with some weight λ.


### 2. Spectral Normalization
The spectral normalization of the _convolutional_ layers is implemented with the customized function [`spectral_norm_conv()`](./src/model/utils/parametrization.py#L94). It is a straight adaptation of the function [`spectral_norm()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html) in PyTorch that leverages the [`register_parametrization()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrize.register_parametrization.html) functionality―thus they have almost identical APIs―except that it does not naïvely normalize the weight/kernel of each linear layer, but instead normalizes the linear map each layer represents. Unlike in the fully-connected layers, the linear map each convolutional layer stands for is dependent on the shape of its input, thus the naïve [`spectral_norm()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html) cannot guarantee the normalized convolutional layers to be 1-Lipschitz. 
More information can be found in this paper: [Regularisation of neural networks by enforcing Lipschitz continuity](https://doi.org/10.1007/s10994-020-05929-w).

Therefore, we should spectral-normalize the convolutional layers and the fully-connected layers in different manners:
* For convolutional layers, [`spectral_norm_conv()`](./src/model/utils/parametrization.py#L94) takes one additional positional argument `in_shape` to incorporate this shape dependence when performing the power method. For a convolutional layer of input shape (N,C,A,B,...), `in_shape` takes (A,B,...) after the channel dimension C as its input.   
This function supports all Conv and ConvTranspose layers in PyTorch.
* For fully-connected layers, [`spectral_norm()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.parametrizations.spectral_norm.html) from PyTorch works fine.

Due to the similarity in their APIs, you may refer to PyTorch's [Parametrizations Tutorial](https://pytorch.org/tutorials/intermediate/parametrizations.html) for more details, or just check out how they are employed in [layers.py](./src/model/layers.py#L80). If you are interested in [`spectral_norm_conv()`](./src/model/utils/parametrization.py#L94)'s normalization accuracy, you may retrieve the convolutional layers' linear maps with [`torch.autograd.functional.jacobian()`](https://pytorch.org/docs/stable/generated/torch.autograd.functional.jacobian.html) and investigate their spectral norms with [`torch.linalg.svdvals()`](https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html).

## Tutorials

Some illustrative low dimensional examples can be found in the Jupyter notebook [least_volume.ipynb](./notebook/least_volume.ipynb).

## Experiments
Run experiments in this paper with the command below under [src/](./src/):
```
python least_volume_image.py \
    -n vol \
    -l 1e-3 \
    -e 400 \
    -b 100 \
    -d cuda:0 \
    --eps 1 \ 
    mnist_vol
```
Information about these arguments can be found via:
```
python train_least_volume.py --help
```
_\* For now it seems CelebA is removed from torchvision (perhaps temporarily) due to some unknown issue, so we commented out all the codes related to CelebA._


## Citation
```
@inproceedings{qiuyi2024compressing,
  title={Compressing Latent Space via Least Volume},
  author={Chen, Qiuyi and Fuge, Mark},
  booktitle={ICLR},
  year={2024}
}
```