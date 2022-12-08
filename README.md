
# Breed Identification System 

Breed Identification System is designed to predict the class of a species of animal. For this project we have chosen Dog, Cat and Fish dataset through open-source commmunities. All datasets were trained on 3 CNN architectures MobileNetV2, ResNet18 and ResNet50, and as result we got 9 instances from scratch, 6 instances for transfer learning and multiple instances for hyperparameter tuning. In this report we discuss different outcomes we got from training (scratch and transfer), hyperparameter tuning, TSNE visualization and what we can infer from all the experimentation done in course of this project. 


## Dataset Information 
Links to the dataset can be found here and the download process is straight forward, if you have a kaggle account and Data_split_preparation contains the curation codes.
https://www.kaggle.com/competitions/dog-breed-identification/data 

https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset

https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

## Train
All models related files attached are in the form of notebooks with imports and pip install commands in place, and the datasets link can be found below to run your own training instance, data modifier codes can also be in the Data_preparation folder to help setup your data.

## Testing
As per the requirements of the project, we have created a sample test data comprising of 100 images, it can be found here along with a notebook for running on a sample test the only change required there is the path updated for validation data and the model path.   
https://drive.google.com/drive/folders/18EB6twUQmhNFn2d5ODOKANAVaWV67Crf?usp=sharing


## Install
Requirements file with all the packages used during development.
```sh
pip install -r requirements.txt
```

## Folder Structure

'MobileNetV2', 'ResNet18' and 'ResNet50' folders contain training from scratch notebooks code for each dataset .

'Transfer_Learning' folder contains notebooks for transfer learning of dog and fish dataset on all the CNN architectures.

'Hyperparameter Tuning' folder contains hyperparamerter tuning related notebooks for learning rate tuning (and a seperate folder for Loss function) on cat dataset using ResNet18.

't-SNE' folder contains t-SNE plot notebooks for fish and dog dataset on all the models.

'Documents' folder contains all the documents prepared during the course of this project.

'final_file.ipnyb' is a notebook for classifying all 3 animal species (dog, cat and fish).

There are some other folders also containing supporting code for normalization, data split and model definitions.


## Authors
Abhishek Amola 40105405

Vishvesh Khandpur 40201421

Nanda Kumar K 40220298

Tannavi Gaurav 40221686

