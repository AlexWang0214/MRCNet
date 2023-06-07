# MRCNet

## Installation
---
we run the project on CUDA11.3
#### **Step 1.** Create a conda virtual environment and activate it
```
conda create -n mrcnet python=3.7 -y
conda activate mrcnet
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

#### **Step 2.** Install MRCNet
```
git clone https://github.com/AlexWang0214/MRCNet.git
cd MRCNet
pip install -r requirements.txt
```

#### **Step 3.** Install Pytorch-Correlation-extension
```
cd ..
git clone https://github.com/ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python setup.py install
```

#### **Step 4.** Download the weights from https://drive.google.com/drive/folders/1zGGxM3EkNNrbpbeQONfA8UlWVrVPUatp?usp=sharing

#### Data Preparation
The data folders are organized as follows:
```
├── data/
|   └── sequences
|       └── 00  
|           └── image_2
|               └── 000000.png
|               └── 000001.png
|               └── ...
|           └──velodyne
|               └── 000000.bin
|               └── 000001.bin
|               └── ...
```


## Testing
Change the 'data_folder', 'output' and 'weights' in `evaluate_calib.py` before running:
```
python evaluate_calib.py
```

# Contact
For questions about our paper or code, please contact alexwang@buaa.edu.cn.
