import torch
import dlutils
import numpy as np
from net import StyleGANInspiredNet
from dataloading import process, get_data
from loss_tracker import LossTracker
from utils import run
import os


def iteration(logger, train_data, validation_data, model, optimizer, criterion, tracker):
    np.random.shuffle(train_data)
    train_batches = dlutils.batch_provider(train_data, batch_size=128, process_func=process)

    model.train()
    for x, y in train_batches:
        y_pred = model(x)

        loss = criterion(y_pred, y)

        tracker.update(dict(train_loss=torch.sqrt(loss.item())))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_batches = dlutils.batch_provider(validation_data, batch_size=1024, process_func=process)

        for x, y in validation_batches:
            y_pred = model(x)

            loss = criterion(y_pred, y)

            tracker.update(dict(validation_loss=torch.sqrt(loss.item())))


def training(cfg, logger):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    np.random.seed(0)

    # Get training and validation data
    train_data, validation_data = get_data(cfg, logger)

    # Initialize the StyleGAN-inspired model and move it to GPU if available
    model = StyleGANInspiredNet(cfg).cuda()

    # Define the Mean Squared Error loss function and Adam optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.BASE_LEARNING_RATE)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.LEARNING_DECAY_STEPS, gamma=cfg.TRAIN.LEARNING_DECAY_RATE)

    # Loss tracker to monitor training progress
    tracker = LossTracker(cfg.OUTPUT_DIR)

    for epoch in range(cfg.TRAIN.EPOCHS):
        # Perform one iteration of training and validation
        iteration(logger, train_data, validation_data, model, optimizer, criterion, tracker)

        # Log training progress and metrics
        logger.info('[%d/%d] - %s, lr: %.12f, max mem: %.2f MB,'
                    % ((epoch + 1), cfg.TRAIN.EPOCHS, str(tracker),
                       optimizer.param_groups[0]['lr'],
                       torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)))

        # Register the mean values for plotting
        tracker.register_means(epoch)

        # Plot the loss curves
        tracker.plot()

        # Step the learning rate scheduler
        scheduler.step()

    # Save the trained model
    model_save_path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_SAVE_FILENAME)
    torch.save(model.state_dict(), model_save_path)

    logger.info("Training completed. Model saved to: %s" % model_save_path)


if __name__ == "__main__":
    run(training)
