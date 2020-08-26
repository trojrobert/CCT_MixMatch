import os 
from dataloaders import voc

if __name__=='__main__':
    working_dir = os.getcwd()
    dataset = voc.VOCDataset(working_dir, "2012")
    dataset.download_voc()