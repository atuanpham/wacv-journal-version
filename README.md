# 3D-Brain Segmentation Using Deep Neural Network and Gaussian Mixture Model

## Abstract:
Automatic segmentation of major brain tissues from high-resolution magnetic resonance images (MRIs) plays an important role in clinical diagnostics and neuroscience research. In this paper, we present a novel approach to extract brain tissues including gray matter, white matter and cerebrospinal fluid by using Gaussian mixture models (GMMs), Convolution neural networks (CNNs) and Deep neural networks (DNNs). GMMs are applied to classify voxels which have distinct intensity information and are easy to recognize while DNNs and CNNs are treating voxels which are similar in appearance and usually recognized insufficiently by traditional approaches. The empirical results on IBSR 18 dataset show that the proposed method outperforms 13 state-of-the-art algorithms, surpassing all other methods by a significant margin.

- link: [http://ieeexplore.ieee.org/document/7926679](http://ieeexplore.ieee.org/document/7926679/)

## Requirements:
- python3
- conda
- cuda

## Data:
- [Data](https://www.nitrc.org/projects/ibsr) used in this project can be downloaded from:
[https://www.nitrc.org/projects/ibsr](https://www.nitrc.org/projects/ibsr)

- After getting data, You may want to consider splitting it into two parts and put to **data/raw/test** and
  **data/raw/train** directories respectively.

## Initiate environment:
- If you would like to use CPU for training model, run command:
```
conda env create
```
- For initiating with GPU support:
```
conda env create -f environment-gpu.yml
```

## Activate environment:
- Environment could be activated by running command `source activate _env-name_`.
- Environment name is based on which command you executed for initiating before (**journal** or **journal-gpu**).

## Usage:
- Before training model, you need to preprocess data by following command:
```
python main.py preprocess
```

- Train model:
```
python main.py train
```

- After model training process is completed, make sure that **unet.hdf5** exists in directory **results/weights**. Run below
  command to predict data:
```
python main.py predict --data-path /path/to/data --predictions-path /path/to/prediction/results.npy
```

- Run `python main.py evaluate` to print score and accuracy based on test data.

## Results
Weights of the trained-model could be get at: [http://bit.ly/2elpZzL](http://bit.ly/2elpZzL).

Before predicting image, You must copy **unet.hdf5** to **results/weights** directory.
