# Details

## Dataset

**TLDR**: Modalities used: 2D Axial MRIs extracted from 3D 1.5T MRIs, associated clinical metadata

In this work the normalized collection `ADNI1:Complete 1Yr 1.5T` is used which includes 2294 1.5T MRI scans.
Moreover, the current model uses axial orthogonal brain slices, accompanied by metadata useful for disease detection that including `diagnosis`, `age`, Mini-Mental State Examination (`MMSE`), Clinical Dementia Rating (`CDR`), regional brain volumes (`ROIs`), `biomarkers`, and `neuropsychological summary scores`.

> [!NOTE] 
> This repository does not include the ADNI data. Access requires registration and approval at https://adni.loni.usc.edu/.

### Metadata Preprocessing
In the case of the metadata, features such as ROI volumes were preprocessed to remove scale differences via standardization, while the ratio of biomarker data, which is often used as an indication of the existence of the disease, was calculated.
 
### MRI Preprocessing
Before extracting the axial slices from the MRI scans, a series of processing methods were applied with the main goals of spatial normalization and removal of the skull and tissues, which were are not necessary in the current study.

![](./Images/3D_Preprocessing.PNG)

After extracting 2D orthogonal axial slices, additional preprocessing was applied with the primary goal of reducing noise, as is shown in the image below:

<p align="center">
  <img src="./Images/2D_Preprocessing.PNG" alt="Description"/>
</p>



## Model Architecture
**TLDR**: Parallel branches for 2D MRI (`CNN`) and metadata (`ANN`)

The CNN sub-network receives 2D MRIs and mainly consists of a pre-trained model used for transfer learning, that extracts the same ammount of features as the ANN sub-network.
Many pre-trained models were used including `ResNet50`, `ResNet50V2`, `VGG-16`, `VGG-19`, `InceptionV3` and `DenseNet121`, in order to identify the most suitable model among them for extracting features from the MRIs. 
At the same time, the ANN sub-network receives as input 20 metadata of the corresponding MRIs, and lastly all features from both networks are concatenated and used in the final ANN network that performs the classification. More details are given below:

- **Training Strategy**: `5-Fold Cross-Validation` with `train-val-test spliting` at subject level to avoid data leakage
- **Loss function**: `Binary Crossentropy` for binary classification, `Sparse Categorical Crossentropy` for multi-class classification
- **Optimizer**: `SGD with Nesterov Momentum`
- **Evaluation Metrics**: `Accuracy`, `Precision`, `Recall`, `F1-score` (used mainly due to data imbalance)

![](./Images/Model.PNG)


## Results

The best performance is found to be for distinguishing between the `CN` and `AD` groups, while at the same time a decrease in performance was observed in models in which observations of the `MCI` group are used.
Among the pretrained models `InceptionV3` is found to have the best average performance for the `CN vs MCI vs AD` and `CN vs AD` models, while for `CN vs MCI` and `MCI vs AD` models `DenseNet121` and `ResNet50V2`, respectively.

### Accuracies
![](./Images/Accuracies.PNG)

### CN vs MCI vs AD
![](./Images/CNvsMCIvsAD.PNG)

### CN vs AD
![](./Images/CNvsAD.PNG)

### CN vs MCI
![](./Images/CNvsMCI.PNG)

### MCI vs AD
![](./Images/MCIvsAD.PNG)

