import os 

def dir_exists(path):
    if not os.path.exists(path):
            os.makedirs(path)