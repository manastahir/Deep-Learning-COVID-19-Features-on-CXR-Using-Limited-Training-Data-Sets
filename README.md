# Deep-Learning-COVID-19-Features-on-CXR-Using-Limited-Training-Data-Sets
Given the limited dataset of Covid-19 CXRs this project aims to combine segmentation and classification models to learn important features and avoid overfitting on limited data.

The repository provides the un-official implementaion of the paper <a href='https://ieeexplore.ieee.org/document/9090149'> <b>Deep Learning COVID-19 Features on CXR Using Limited Training Data 
Sets</b></a>.

<img src="https://github.com/manastahir/Deep-Learning-COVID-19-Features-on-CXR-Using-Limited-Training-Data-Sets/blob/master/architechure.png" height="400" width="800" alt="architechure">
<hr/>

## Introduction
The data for COVID-19 CXR is very limited and appliying deep learning models to the limited datasets tends to lead to overfitting, in order to prevent the overfitting and learn relevent 
features the segmentation network was used to produce the segmentation mask for CXR. After applying this mask on the image N random patches of size 224x224 are 
extrcacted. The center of the patch has to lie inside the Lung region.The patches are then fed to a classifier and majority vote is taken to do the final classification. In the 
paper they proposed to use N=100. I ran the experiments on free colab and due to limited resources I had to use N=25. Due to this results were slightly worse than in the paper, but 
acceptable.

Segmentaion network: DenseNet103
Classification network: pretrained ResNet18

## Requirements
```shell
pip -q install requirements.txt
```

## Data Collection and Processing

Data is collected from several different opensource resources, links are provided:

<a href=https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip>Montgomery Lung Segmentaion Dataset</a><br/>
<a href=http://db.jsrt.or.jp/data_download_contents/download_data/All247images.zip>JSRT/SCR Lung Segmentaion Dataset</a><br/>
<a href=https://github.com/ieee8023/covid-chestxray-dataset/archive/master.zip>COVID-19 CXR</a><br/>
<a href=https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset>Coronahack dataset</a><br/>

Data processing is done to processing the segmentaion masks, combine the images from all the different datasets and split the images according the ratio mention in the paper.
