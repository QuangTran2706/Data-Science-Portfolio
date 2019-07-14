import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch.nn.functional as F

import json
import argparse

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Load a model from a checkpoint and make a prediction on an image")
    parser.add_argument('--img_path', dest='image_path', default='./flowers/test/13/image_05767.jpg', type=str, help='set the image path')
    parser.add_argument('--topk', dest='topk', default=5, type=int, help='set the num of topk')
    parser.add_argument('--gpu_mode',dest='gpu_mode', default=True, type=bool, help='set the gpu mode')
    parser.add_argument('--ckpt_path', dest='check_point_path', default='checkpoint.pth', type=str, help='set the checkpoint path')
    parser.add_argument('--jsonfile_path',dest='json_file_path', default='cat_to_name.json', type=str, help='set a json file path')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    """a function that loads a checkpoint and rebuilds the model"""
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
           
    pil_image = Image.open(image)
    
    transformer = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    pil_image = transformer(pil_image)
    return pil_image.numpy()

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    args = parse_args()
    json_file = args.json_file_path
    with open(json_file, 'r') as f:
    		class_name = json.load(f)
# open and normalize image and convert numpy array back to tensor
    image = torch.from_numpy(process_image(image_path))
    # turn on eval mode
    cuda = torch.cuda.is_available()
    if gpu and cuda:
        model.cuda()
    else:
        model.cpu()

    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
    
    ps = torch.exp(output).data.topk(topk)
    probs = ps[0].squeeze().tolist()
    class_index = ps[1].squeeze().tolist()
    # cat_to_name maps class number to flower name, model.class_to_idx maps class number to class index
    # now that I have the class_index, I need to invert the class_to_idx dictionary 
    # to use index as key and class number as the value. with that inverted dictionary, I can map back to flower name.
    index_to_class = {k:v for v,k in model.class_to_idx.items()}
    # now class index is the key, class number is the value, dictionary is inverted
    top_classes = [index_to_class[k] for k in class_index]
    # map to flower name with the class number 
    class_names = [class_name[i] for i in top_classes]
    return probs, top_classes, class_names

def main():
	args= parse_args()
	filepath = args.check_point_path
	gpu = args.gpu_mode
	image_path= args.image_path
	topk= args.topk

	model = load_checkpoint(filepath)
	probs, top_classes, class_names = predict(image_path, model, topk, gpu)
	print('Predicted Most Likely Class: {}, {} with probability of {:.4f}'.format(class_names[0], top_classes[0], probs[0]))
	print('Predicted Top {} Classes and corresponding probabilities:'.format(topk))
	print(class_names)
	print(probs)

if __name__ == '__main__':
    main()
