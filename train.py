import torch
import dlutils
import random
from net import StyleGANInspiredNet
from dataloading import process, get_data
from loss_tracker import LossTracker
from utils import run
import os


def iteration(logger, train, validation, model, optimizer, criterion, tracker):
    random.shuffle(train)
    batches = dlutils.batch_provider(train, 128, process)

    model.train()
    for x, y in batches:
        y_pred = model(x)

        loss = criterion(y_pred, y)

        tracker.update(dict(train_loss=torch.sqrt(loss)))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        batches = dlutils.batch_provider(validation, 1024, process)

        for x, y in batches:
            y_pred = model(x)

            loss = criterion(y_pred, y)

            tracker.update(dict(validation_loss=torch.sqrt(loss)))


def training(cfg, logger):
    train, validation = get_data(cfg, logger)

    model = StyleGANInspiredNet(cfg).cuda()

    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.BASE_LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.LEARNING_DECAY_STEPS, gamma=cfg.TRAIN.LEARNING_DECAY_RATE)
    tracker = LossTracker(cfg.OUTPUT_DIR)

    for i in range(cfg.TRAIN.EPOCHS):
        iteration(logger, train, validation, model, optimizer, criterion, tracker)
        logger.info('[%d/%d] -  %s, lr: %.12f, max mem: %f",' % (
            (i + 1), cfg.TRAIN.EPOCHS, str(tracker),
            optimizer.param_groups[0]['lr'],
            torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
        tracker.register_means(i)
        tracker.plot()
        scheduler.step()

    torch.save(model, os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_SAVE_FILENAME))


if __name__ == "__main__":
    run(training)
