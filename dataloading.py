import numpy as np
import torch
import pickle


def process(batch):
    x = [x[0] for x in batch]
    y = [x[1] for x in batch]
    x = np.asarray(x)
    x = np.pad(x, ((0, 0), (0, 0), (0, 3)), mode='constant')
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def load_data(file_x, file_y):
    with open(file_x, 'rb') as f:
        train_X = pickle.load(f)
    with open(file_y, 'rb') as f:
        train_Y = pickle.load(f)

    return list(zip(train_X, train_Y))


def get_data(cfg, logger):
    train = load_data('train_X.pkl', 'train_Y.pkl')
    validation = load_data('validation_X.pkl', 'validation_Y.pkl')

    logger.info("Mean of X %f" % np.asarray([x[0] for x in train]).mean())
    logger.info("Std of X %f" % np.asarray([x[0] for x in train]).std())

    logger.info("Mean of Y %f" % np.asarray([x[1] for x in train]).mean(axis=0))
    logger.info("Std of Y %f" % np.asarray([x[1] for x in train]).std(axis=0))

    return train, validation
