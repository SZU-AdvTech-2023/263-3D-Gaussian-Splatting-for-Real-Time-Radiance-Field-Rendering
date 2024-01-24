# Gaussion_splatting

## Installation

```
# create virtual environment
conda create -yn gs python=3.9 pip
conda activate gs

# install the PyTorch first, you must install the one match to the version of your nvcc (nvcc --version)
# for cuda 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# install other requirements
pip install -r requirements.txt
```
## Download the Blender dataset and unzip it into the 'dataset' directory.

## Train
```
python train.py
```

## Evaluation

```
python eval.py 
```
## Render a video.
```
python render.py 
```

