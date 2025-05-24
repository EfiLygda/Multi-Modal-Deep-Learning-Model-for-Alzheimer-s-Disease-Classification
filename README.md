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
