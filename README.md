# Multi Modal Deep Learning Model for Alzheimer's Disease Classification

This repository contains the scripts used for the architecture and evaluation of a multimodal deep learning model in order to distinguish between three cognitive states of Alzheimer’s Disease. 
Specifically using data from participants in the ADNI database, the disease’s states are distinguished into Cognitively Normal (`CN`), Mild Cognitive Impaired (`MCI`) and Mild Alzheimer’s Disease
(`AD`). 
The data used, to train the model, included pre-processed 2D axial MRI scans, as well as supplementary metadata. 
The model consists of two sub-networks, extracting the same number of features from the two kinds of data, that are connected to a final network, which performs the categorization of the observations between the three diagnostic groups. 
Specifically, the first sub-network extracts features from the axial slices, using one of the pre-trained CNN models, among `VGG-16`, `VGG-19`, `ResNet50`, `ResNet50V2`, `InceptionV3` and `DenseNet121`, while the second one extracts features from the 20 additional metadata, through a simple artificial neural architecture.

Then, the architecture of the model was evaluated based on its ability to distinguish between all three diagnostic groups, as well as their two-by-two combinations, as well as the performance of the 6 pre-trained CNN models. 
Finally, the best performance is found to be for distinguishing between the `CN` and `AD` groups, while at the same time a decrease in performance was observed in models in which observations of the `MCI` group are used.
Among the pretrained models `InceptionV3` is found to have the best average performance for the `CN vs MCI vs AD` and `CN vs AD` models, while for `CN vs MCI` and `MCI vs AD` models `DenseNet121` and `ResNet50V2`, respectively.

![](./Images/Model.PNG)

> [!NOTE] 
> This repository does not include the ADNI data. Access requires registration and approval at https://adni.loni.usc.edu/.


## Requirements

### Python Requirements
```
python==3.8.18
numpy==1.19.5
pandas==1.3.4
scikit-learn==1.3.2
tensorflow==2.3.0
matplotlib==3.5.1
opencv-python==4.8.1.78
scikit-image==0.19.3
nibabel==5.1.0
tqdm==4.66.1
```

### Cuda Requirements
```
Cuda==10.1.243
cuDNN==7.6.0.64
```

### Linux System Requirements (WSL) Requirements
```
ANTs==2.5.0.post14-gfe3a0e3
FSL==6.0.7.3
FSLeyes==1.11.0
```

## Repository Structure

* `csv`: Should contain the tables from ADNI containg the MRI metadata.
* `Images`: Contains the images used in the ReadMe.md file
* `DataCleaning`: Contains the scripts used for merging and cleaning the metadata, resulting in csv file used for training the model.
* `PrepareMRIs`: Contains the scripts used for extracting, preprocessing and labeling the MRIs.
* `Model`: Contains the scripts used for configuring and training the model.
* `Details.md`: Contains a brief outline of the preprocessing used for the MRIs, the training of the model and lastly the results.

## License
This project is licensed under the terms of the `Apache-2.0 license`.

## Status
> Done
