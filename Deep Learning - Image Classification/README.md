### Project: Image Classification


### Data

The dataset contains 102 different types of flowers. It is split into train, validate and test sets. 
There are about 20 images per flower in the training set.  
The data for this project is quite large - in fact, it is so large it cannot be uploaded onto Github. 

### Project Overview
This project uses Pytorch to build a feedforward neural network to train the flower images with tranfer learning. 
Then the trained classifier is used to predict the species for new images of the flowers.

### Application
The Jupyter Notebook is converted into two python files - 
The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. The second file, predict.py, uses a trained network to predict the flower category with the highest probability for an input image.

#### Instructions to the .py files locally:
1. train.py
Basic run with all default arguments: python train.py
Optional arguments:
* set data folder path: python train.py --data_dir "../flower_data"
* choose pretrained network: python train.py --arch "vgg16"
* set learning rate: python train.py --learning_rate 0.001
* set hidden units: python train.py --hidden_units [2048, 1024]
* choose epochs: python train.py --epochs 15
* whether to train with GPU mode: python train.py --gpu True


2. predict.py
Basic run with all default arguments: python predict.py
Optional arguments:
* set test image file path: python predict.py --img_path "flower_data/test/13/image_05767.jpg"
* set the checkpoint file path: python predict.py --ckpt_path 'checkpoint.pth'
* choose top k classes of the flower image: python predict.py --topk 10
* set the flower number to name mapping dictionary path: python predict.py --jsonfile_path "cat_to_name.json"



