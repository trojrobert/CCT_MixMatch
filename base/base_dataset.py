import numpy as np 
import random
import cv2 
from PIL import Image 
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BaseDataSet(Dataset):

    def __init__(self, data_dir, split, splits_dir, mean, std, ignore_index, base_size=None, augment=True, colour_attributes=None, val=False,
                jitter=False, use_weak_lables=False, weak_labels_output=None, crop_size=None,
                 scale=False, flip=False, rotate=False, blur=False, return_id=False, n_labeled_examples=None):

        self.root =  data_dir
        self.splits_dir = splits_dir
        self.split = split

        self.mean = mean 
        self.std = std

        self.crop_size = crop_size
        self.colour_attributes = colour_attributes
        self.ignore_index = ignore_index
        self.val = val
        self.jitter = jitter


        self.image_padding = (np.array(self.mean)*255.).tolist()

        self.n_labeled_examples = n_labeled_examples

        self.return_id = return_id

        self.augment = augment
        if self.augment:
            self.base_size = base_size
            self.scale = scale 
            self.flip = flip
            self.rotate = rotate
            self.blur = blur

       

        self.use_weak_lables = use_weak_lables
        self.weak_labels_output = weak_labels_output

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

        self.files = []
        self._set_files()

    """
    def _augmentation(self, index, image, label):
        
        if self.val: 
            image, label = self._val_augmentation(image, label)

        elif self.augment:
            image, label = self._train_augmentation(image, label)
        
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        return image, label 
    """
    def __len__(self):
        """change the behaviour of len method"""
        return len(self.files)

    def __repr__(self):
        """change the behaviour of the print function"""

        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

    def __getitem__(self, index):
        """load and perform data argumentation
        
        Args:
            index (): 
        
        Returns:
            image ():
            label ()
        """
        image, label, image_id = self._load_data(index)

        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment: 
            image, label = self._augmentation(image, label)

        #convert numpy array to tensor
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()

        return image, label 

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        """resize validate data

        Args:
            image (): raw image
            label (): raw label image

        Returns:
            image (): resized image  
            label (): resized label 
        """

        if self.base_size is not None:
            image, label = self._resize(image, label)
        image = self.normalize(self.to_tensor(Image.fromarray(np.unit8(image))))
        return image, label

    def _augmentation(self, image, label):
        """transform training data

        Args:
            image (): image to transform
            label (): label of image

        Returns:
            image (): transformed image  
            label (): transformed label image 
        """

        if self.base_size is not None:
            image, label = self._resize(image, label)

        if self.crop_size is not None:
            image, label = self._crop(image, label)

        if self.flip: 
            image, label = self._flip(image, label)

        if self.jitter:
            image , label = self._jitter(image, label)
        
        image = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image))))
        return image, label


    def _resize(self, image, label, bigger_side_to_base_size=True):
        """resize  image to basse size 

        Args:
            image (): raw image
            label (): raw label image

        Returns:
            image (): resized image
            image (): resized label image
        """

        if isinstance(self.base_size, int):

            h, w, _ = image.shape
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
                #longside = random.randint(int(self.base_size*0.5), int(self.base_size*1))
            else:
                longside = self.base_size

            if bigger_side_to_base_size:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            else:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h < w else (int(1.0 * longside * h / w + 0.5), longside)
            image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            return image, label



    def _crop(self, image, label):
        """crop image 

        Args:
            image (): image to transform
            label (): label of image

        Returns:
            image (): cropped image  
            label (): label of cropped image 
        """
        # Padding to return the correct crop size
        if (isinstance(self.crop_size, list) or isinstance(self.crop_size, tuple)) and len(self.crop_size) == 2:
            crop_h, crop_w = self.crop_size 
        elif isinstance(self.crop_size, int):
            crop_h, crop_w = self.crop_size, self.crop_size 
        else:
            raise ValueError

        h, w, _ = image.shape
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,}
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_index, **pad_kwargs)

        # Cropping 
        h, w, _ = image.shape
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        return image, label

        return image, label 

    def _flip(self, image, label):

        """flip image 

        Args:
            image (): image to transform
            label (): label of image

        Returns:
            image (): flip image  
            label (): lable of flip image 
        """
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()

        return image, label 


    def _jitter(self, image, label):

        """flip image 

        Args:
            image (): image to transform
            label (): label of image
            color_attributes (dict): attributes for colour transformation 

        Returns:
            image (): jitter image  
            label (): lable of jitter image 
        """
        attributes = list(self.color_attributes.keys())
        expected_attributes = ["brightness", "contrast", "saturation", "hue"]
        if expected_attributes == intersection(attributes, expected_attributes):

            image = transforms.ColorJitter(
                brightness=color_attributes.brightness,
                contrast=colour_attributes.contrast,
                saturation=color_attributes.saturation,
                hue=color_attributes.hue
            )
        return image, label

