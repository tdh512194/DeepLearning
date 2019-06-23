import torchvision

def transform_preprocess(height, width):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.ToTensor()
    ])
    return transform

def is_correct():
    pass