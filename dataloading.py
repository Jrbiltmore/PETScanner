import numpy as np
import torch
import pickle


def process(batch):
    x = [x[0] for x in batch]
    y = [x[1] for x in batch]
    x = np.asarray(x)
    #x = np.pad(x, ((0, 0), (0, 0), (0, 3)), mode='constant')
    x = np.pad(x, ((0, 0), (0, 4), (0, 0)), mode='constant')
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data['train'], data['validation']


def get_data(cfg, logger):
    train, validation = load_data(cfg.DATASET.PATH)
    logger.info("Number of train images: %d" % len(train))
    logger.info("Number of validation images: %d" % len(validation))

    properties = dict(
        x_mean=np.asarray([x[0] for x in train]).mean(),
        x_std=np.asarray([x[0] for x in train]).std(),
        y_mean=np.asarray([x[1] for x in train]).mean(axis=0),
        y_std=np.asarray([x[1] for x in train]).std(axis=0),
    )

    logger.info(
        """\n"""
        """Mean of X {x_mean}\n"""
        """Std of X {x_std}\n"""
        """Mean of Y {y_mean}\n"""
        """Std of Y {y_std}\n"""
        .format(**properties))

    return train, validation
