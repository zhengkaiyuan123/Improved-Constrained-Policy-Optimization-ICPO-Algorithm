influential simplices mining neural network (ISMnet) that achieves SOTA performance in this task.
<div align="center">
    <img src="figs/framework.png" alt="Framework">
</div>



# Installation
This code is developed with Python3, and we recommend python>=3.8 and PyTorch ==1.13.0. Install the dependencies with Anaconda and activate the environment with:

    conda create --name GOUB python=3.8
    conda activate GOUB
    pip install -r requirements.txt



# Important Option Details
* `dataroot_GT`: Ground Truth (High-Quality) data path.
* `dataroot_LQ`: Low-Quality data path.
* `pretrain_model_G`: Pretraind model path.
* `GT_size, LQ_size`: Size of the data cropped during training.
* `niter`: Total training iterations.
* `val_freq`: Frequency of validation during training.
* `save_checkpoint_freq`: Frequency of saving checkpoint during training.
* `gpu_ids`: In multi-GPU training, GPU ids are separated by commas in multi-gpu training.
* `batch_size`: In multi-GPU training, must satisfy relation: *batch_size/num_gpu>1*.


# Citation
Please cite our work if you find our code/paper is useful to your work. :
```latex
@article{ISMnet2024, 
    title = {Influential simplices mining via simplicial convolutional networks}, 
    journal = {Information Processing \& Management}, 
    author = {Yujie Zeng and Yiming Huang and Qiang Wu and Linyuan L{\"u}}, 
    volume = {61}, 
    number = {5}, 
    pages = {103813}, 
    year = {2024}, 
    issn = {0306-4573}, 
    doi = {https://doi.org/10.1016/j.ipm.2024.103813}, 
    url = {https://www.sciencedirect.com/science/article/pii/S0306457324001729}, 
}
```


 
Thank you for your interest in our work. If you have any questions or encounter any issues while using our code, please don't hesitate to raise an issue or reach out to us directly.
