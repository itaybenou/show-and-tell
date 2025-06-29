import torch
from torchvision import datasets, transforms
from transformers import ResNetForImageClassification


DATASET_ROOTS = {"imagenet_sample_val": "demos/images/imagenet_val_samples"}
LABEL_FILES = {"imagenet":"data/imagenet_classes.txt"}


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_sample_data(dataset_name, preprocess=None):
    if dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)   
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized.")
    return data


def get_target_model(target_name, device):
    
    if target_name == 'resnet50_imagenet':
        target_model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    else:
        raise ValueError(f"Target model {target_name} not recognized.")
    
    return target_model, preprocess