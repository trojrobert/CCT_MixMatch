import os 
import numpy as np 
from PIL import Image 

from base import BaseDataSet, BaseDataLoader

from torchvision import datasets

class VOCDataset(BaseDataSet):
    def __init__(self, **kwargs):

        super(VOCDataset, self).__init__(**kwargs)
  
        
    def _load_data(self, index):
        """Create batch data based on index 

        Args:
            index (int): index of data

        Returns:
            image (): image specific to the index 
            label (): label specific to the index
            image_id (): image_id specific to the index 
        """


        image_path = os.path.join(self.root, self.files[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]

        label_path = os.path.join(self.root, self.labels[index][1:])
        label = np.asarray(Image.open(label_path), dtype=np.int32)

        return image, label, image_id

    def _set_files(self):
        """Using a custom txt file to create image and label path names"""

        self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')

        if self.split == 'val':
            file_list = os.path.join(self.splits_dir, f"{self.split}" + ".txt")

        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(self.splits_dir, f"{self.n_labeled_examples}_{self.split}" + ".txt")

        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

class VOC(BaseDataLoader):
    """What do you do"""

    def __init__(self, kwargs):
        self.batch_size = kwargs.pop('batch_size')

        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        #Data augmentation
        self.dataset = VOCDataset(**kwargs)

        super(VOC, self).__init__(self.dataset, self.batch_size, shuffle, num_workers, val_split=None)

class Downloader:
    """Download data"""

    def __init__(self, root, year):
        self.root = root
        self.year = year
        self._download_voc()


    def _download_voc(self):
        """Download VOC Dataset using pytorch"""

       
        if not os.path.exists(self.root ):
            os.makedirs(self.root )
            datasets.VOCSegmentation(self.root, year=self.year, image_set="train", download=True)
    
    
    
