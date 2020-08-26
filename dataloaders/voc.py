import os 

from PIL import Image 


from torchvision import datasets

class VOCDataset:
    def __init__(self, data_path, year, split):
        self.data_path = data_path
        self.year = year 
        self.split = split

        self.download_voc()

    def download_voc(self):
        """Download VOC Dataset using pytorch"""

        self.data_path =  self.data_path  + "/data"
        if not os.path.exists(self.data_path ):
            os.makedirs(self.data_path )
        
        base_dataset = datasets.VOCSegmentation(self.data_path, year=self.year, image_set="train", download=True)


    def set_data_files(self):
        """Using a custom txt file create image and label path names"""

        self.data_path = os.path.join(self.data_path, 'VOCdevkit/VOC2012')

        if self.split == 'val':
            file_list = os.path.join("dataloader/voc_splits", f"{self.split}" + ".txt")

        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join("dataloader/voc_splits", f"{self.n_labeled_examples}_{self.split}" + ".txt")

        else:
            raise ValueError(f"Invalid split name {self.split}")

        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    
    def load_data(self, index):
        """Create dataset based on index 

        Args:
            index (int): index of data

        Returns:
            image (): image specific to the index 
            label (): label specific to the index
            image_id (): image_id specific to the index 
        """

        image_path = os.path.join(self.data_path, self.files[index][1:])
        image = np.asarray(Image.open(image_path). dtype=np.float32)
        image_id = self.files[index].split("/")[-1].split(".")[0]

        label_path = os.path.join(self.data_path, self.labels[index][1:])
        label = np.asarray(Image.open(label_path), dtype=n[.int32])

        return image, label, images_id

    
