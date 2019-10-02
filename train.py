import torch
import math
import dlutils
import random
from net import StyleGANInspiredNet
from dataloading import process, get_data
from loss_tracker import LossTracker
from utils import run
import os


def inference(model, data):
    model.eval()
    pred = []
    with torch.no_grad():
        batches = dlutils.batch_provider(data, 1024, process)

        for x, y in batches:
            y_pred = model(x)
            pred.append(y_pred)

    pred = torch.cat(pred, dim=0)
    return pred


def iteration(logger, train, validation, model, optimizer, criterion, tracker):
    random.shuffle(train)
    batches = dlutils.batch_provider(train, 128, process)

    model.train()
    for x, y in batches:
        y_pred = model(x)

        loss = criterion(y_pred, y)

        tracker.update(dict(train_loss=math.sqrt(loss.item())))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info('loss: %f' % train_loss)

    model.eval()
    with torch.no_grad():
        batches = dlutils.batch_provider(validation, 1024, process)

        for x, y in batches:
            y_pred = model(x)

            loss = criterion(y_pred, y)

            tracker.update(dict(validation_loss=math.sqrt(loss.item())))


def training(cfg, logger):
    train, validation = get_data(cfg, logger)

    model = StyleGANInspiredNet().cuda()

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    tracker = LossTracker(cfg.OUTPUT_DIR)

    for i in range(cfg.TRAIN.EPOCHS):
        iteration(logger, train, validation, model, optimizer, criterion, tracker)
        tracker.register_means(i)
        tracker.plot()

    torch.save(model, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_SAVE_FILENAME))


if __name__ == "__main__":
    run(training)
