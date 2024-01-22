import argparse
import torch
from torch import nn, optim
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import json
import time

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['structure']](pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['state_dict']['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image)
    np_image = transform(image).numpy()
    
    return np_image

def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    model.eval()
    image = torch.from_numpy(image).float()
    image.unsqueeze_(0)
    
    log_ps = model.forward(image)
    ps = torch.exp(log_ps)
    top_p, top_class = ps.topk(topk, dim=1)
    
    return top_p, top_class

def main():
    parser = argparse.ArgumentParser(description='Flower Prediction')
    
    parser.add_argument('--image_path', type=str, help='Image Path')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint Path')
    parser.add_argument('--top_k', type=int, default=3, help='Top K Most Likely Classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real names')
    parser.add_argument('--gpu', type=str, default='gpu', help='Use GPU for training')
    
    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint)
    
    if args.gpu == 'gpu':
        model = model.to('cuda')

    top_p, top_class = predict(args.image_path, model, args.top_k)
    
    top_p = top_p.cpu().numpy().tolist()[0]
    top_class = top_class.cpu().numpy().tolist()[0]
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    labels = [cat_to_name[str(label)] for label in top_class]
    
    print("\nTop K Most Likely Classes and their Probabilities:")
    
    for i in range(args.top_k):
        print(f"Rank {i+1}: {labels[i]} with a probability of {top_p[i]:.3f}")

if __name__ == "__main__":
    main()
