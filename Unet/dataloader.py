import os
from torch.utils.data import Dataset
import pandas as pd
from skimage import io, transform

class SegmentationDataset(Dataset):
    """
    Semantic Segmentation dataset
    X: image 
    Y: masked image
    """

    def __init__(self, csv_path, root_input_dir, root_label_dir, transform=None, label_prefix='label'):
        """
        Args:
            csv_path (string):       path to csv file containing names of imgs
            root_input_dir (string): path to the folder containing the input imgs
            root_label_dir (string): path to the folder containing the label imgs
        """
        self.img_names = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        self.label_prefix = label_prefix

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        base_img_name = self.img_names.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, base_img_name)
        input_img = io.imread(img_name)
        label_name = os.path.join(self.root_label_dir, self.label_prefix, '_', base_img_name)
        label_img = io.imread(label_name)
    
        if self.transform:
            input_img = self.transform(input_img)

        sample = {'input': input_img, 'label': label_img}
        return sample