```
conda create -n parahome python=3.9  
conda activate parahome

# CUDA
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# without CUDA: check https://pytorch.org/get-started/previous-versions/ and install proper version

# Install pytorch3D (It's just for debug visualization.)
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
# CUDA
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt201/download.html  
# without CUDA: check https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

pip install open3d
```
