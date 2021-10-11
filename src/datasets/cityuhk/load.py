import os
import pickle


def load_path_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_datalist(root, mixed=False):
    datalist_file = "scene-mixed" if mixed else "scene-split"
    datalist_file = f"{datalist_file}.datalist"
    datalist_file = os.path.join(root, datalist_file)
    datalist = load_path_data(datalist_file)
    return datalist
