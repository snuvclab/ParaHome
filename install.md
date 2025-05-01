```
conda create -n parahome python=3.9
conda activate parahome

# CUDA
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# without CUDA: check https://pytorch.org/get-started/previous-versions/ and install proper version

pip install open3d
```
