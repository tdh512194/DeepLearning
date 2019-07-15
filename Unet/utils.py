import torchvision


def transform_preprocess(height, width):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((height, width)),
        torchvision.transforms.ToTensor()
    ])
    return transform


def segmentation_correct(preds, labels):
    return 0


def weighted_dice(outputs, targets, weight):
    return 0
