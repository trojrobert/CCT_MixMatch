from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle, num_workers, val_split = 0.0):

        self.shuffle = shuffle
        self.dataset = dataset 
        self.nbr_examples = len(dataset)

        if val_split: 
            #get training and validation indexes 
            #self.train_sampler, self.val_sampler = self._split_sampler(val_split)
            pass
        else:
            self.train_sampler, self.val_sampler = None, None 

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True 
        }    

        super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)

