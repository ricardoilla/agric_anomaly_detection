import json
import numpy as np


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm


def read(features_file, features_len=48):
    with open(features_file) as json_file:
        data = json.load(json_file)
        window_names = data.keys()
        features = data.values()
        d = [[] for x in range(features_len)]
        for feature in features:
            y = 0
            while y < features_len:
                d[y].append(float(feature[y]))
                y += 1
    X = np.array(d)
    X= np.transpose(X)
    X = normalize(X)
    return X, window_names