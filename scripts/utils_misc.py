import os
import pickle

def pickle_load_or_create(path, on_not_exists, config=False):
    """ 
    Path : path to pickle data, 
    on_not_exists : function to be called to initialize new 
    """
    path = path if config else '../data/' + path 
    path += '.pkl'
    print(path)
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return pickle.load(file)
    else:
        return on_not_exists()

def pickle_save(path, data, config=False):
    path = path if config else '../data/' + path 
    path += '.pkl'
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)
