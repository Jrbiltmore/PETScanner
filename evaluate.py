import os
import math
import torch
import seaborn as sns
import dlutils
from dataloading import process, get_data
from matplotlib import pyplot as plt
from utils import run


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


def evaluate(cfg, logger):
    model = torch.load(os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_SAVE_FILENAME))

    logger.info("********* %s test case results*********" % cfg.NAME)
    _, validation = get_data(cfg, logger)

    validation1_predictions = inference(model, validation)

    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(validation1_predictions,
                     torch.tensor([validation[i][1] for i in range(len(validation))], dtype=torch.float32))

    validation1_actual_y = torch.tensor([validation[i][1] for i in range(len(validation))], dtype=torch.float32)

    logger.info("RMSE of the validation set: %f" % math.sqrt(loss))
    errors = abs(validation1_predictions - validation1_actual_y)

    total_errors = []
    for _ in errors:
        total_errors.append(math.sqrt(sum([_[0] ** 2, _[1] ** 2, _[2] ** 2])))

    total_errors = torch.tensor(total_errors)
    total_errors = total_errors.cpu()
    total_errors = total_errors.detach().numpy()

    errors = errors.cpu()
    errors = errors.detach().numpy()

    errors_x, errors_y, errors_z = [], [], []
    for _ in errors:
        errors_x.append(_[0])
        errors_y.append(_[1])
        errors_z.append(_[2])

    errors_x = torch.tensor(errors_x, dtype=torch.float32)
    errors_y = torch.tensor(errors_y, dtype=torch.float32)
    errors_z = torch.tensor(errors_z, dtype=torch.float32)
    errors_x = errors_x.cpu()
    errors_y = errors_y.cpu()
    errors_z = errors_z.cpu()

    logger.info("The mean errors on all coordinates: %s" % str(errors.mean(0)))
    logger.info("The standard deviations on all coordinates: %s" % str(errors.std(0)))

    logger.info("--------------%s Set Absolute Error Results--------------" % cfg.NAME)

    logger.info("****Total Error Stats****")
    logger.info("Mean of the total error: %f" % total_errors.mean())
    logger.info("Standard Deviation of the total error: %f" % total_errors.std())
    logger.info("****X-Coordinate Stats****")
    logger.info("Mean of the error in x-coordinate: %f" % errors_x.mean())
    logger.info("Standard Deviation of the error in x-coordinate: %f" % errors_x.std())
    logger.info("****Y-Coordinate Stats****")
    logger.info("Mean of the error in y-coordinate: %f" % errors_y.mean())
    logger.info("Standard Deviation of the error in y-coordinate: %f" % errors_y.std())
    logger.info("****Z-Coordinate Stats****")
    logger.info("Mean of the error in z-coordinate: %f" % errors_z.mean())
    logger.info("Standard Deviation of the error in z-coordinate: %f" % errors_z.std())

    # print("Total error distribution")
    ax = sns.distplot(total_errors)
    axis_title = ax.set(title='Validation %s Distribution of Total Absolute Error' % cfg.NAME)
    ax.set(xlabel='Normal Distribution M=' + str(total_errors.mean()) + ' SD=' + str(total_errors.std()),
           ylabel='Frequency')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, str(axis_title[0]).split("'")[-2]))
    plt.show()

    ax = sns.distplot(errors_x)
    axis_title = ax.set(title='Validation %s Distribution of X-Coordinate Absolute Errors' % cfg.NAME)
    ax.set(xlabel='Normal Distribution M=' + str(errors_x.mean()) + ' SD=' + str(errors_x.std()), ylabel='Frequency')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, str(axis_title[0]).split("'")[-2]))
    plt.show()

    ax = sns.distplot(errors_y)
    axis_title = ax.set(title='Validation %s Distribution of Y-Coordinate Absolute Errors' % cfg.NAME)
    ax.set(xlabel='Normal Distribution M=' + str(errors_y.mean()) + ' SD=' + str(errors_y.std()), ylabel='Frequency')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, str(axis_title[0]).split("'")[-2]))
    plt.show()

    ax = sns.distplot(errors_z)
    axis_title = ax.set(title='Validation %s Distribution of Z-Coordinate Absolute Errors' % cfg.NAME)
    ax.set(xlabel='Normal Distribution M=' + str(errors_z.mean()) + ' SD=' + str(errors_z.std()), ylabel='Frequency')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, str(axis_title[0]).split("'")[-2]))
    plt.show()

    errors = (validation1_predictions - validation1_actual_y)

    total_errors = []
    for _ in errors:
        total_errors.append(math.sqrt(sum([_[0] ** 2, _[1] ** 2, _[2] ** 2])))

    total_errors = torch.tensor(total_errors)
    total_errors = total_errors.cpu()
    total_errors = total_errors.detach().numpy()

    errors = errors.cpu()
    errors = errors.detach().numpy()

    errors_x, errors_y, errors_z = [], [], []
    for _ in errors:
        errors_x.append(_[0])
        errors_y.append(_[1])
        errors_z.append(_[2])

    errors_x = torch.tensor(errors_x, dtype=torch.float32)
    errors_y = torch.tensor(errors_y, dtype=torch.float32)
    errors_z = torch.tensor(errors_z, dtype=torch.float32)
    errors_x = errors_x.cpu()
    errors_y = errors_y.cpu()
    errors_z = errors_z.cpu()

    logger.info("The mean errors on all coordinates: %s" % str(errors.mean(0)))
    logger.info("The standard deviations on all coordinates: %s" % str(errors.std(0)))

    logger.info("--------------Validation %s Set Results--------------" % cfg.NAME)

    logger.info("****Total Error Stats****")
    logger.info("Mean of the total error: %f" % total_errors.mean())
    logger.info("Standard Deviation of the total error: %f" % total_errors.std())
    logger.info("****X-Coordinate Stats****")
    logger.info("Mean of the error in x-coordinate: %f" % errors_x.mean())
    logger.info("Standard Deviation of the error in x-coordinate: %f" % errors_x.std())
    logger.info("****Y-Coordinate Stats****")
    logger.info("Mean of the error in y-coordinate: %f", errors_y.mean())
    logger.info("Standard Deviation of the error in y-coordinate: %f" % errors_y.std())
    logger.info("****Z-Coordinate Stats****")
    logger.info("Mean of the error in z-coordinate: %f", errors_z.mean())
    logger.info("Standard Deviation of the error in z-coordinate: %f" % errors_z.std())

    # print("Total error distribution")
    ax = sns.distplot(total_errors)
    axis_title = ax.set(title='Validation %s Distribution of Total Error' % cfg.NAME)
    ax.set(xlabel='Normal Distribution M=' + str(total_errors.mean()) + ' SD=' + str(total_errors.std()),
           ylabel='Frequency')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, str(axis_title[0]).split("'")[-2]))
    plt.show()

    ax = sns.distplot(errors_x)
    axis_title = ax.set(title='Validation %s Distribution of X-Coordinate Errors' % cfg.NAME)
    ax.set(xlabel='Normal Distribution M=' + str(errors_x.mean()) + ' SD=' + str(errors_x.std()), ylabel='Frequency')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, str(axis_title[0]).split("'")[-2]))
    plt.show()

    ax = sns.distplot(errors_y)
    axis_title = ax.set(title='Validation %s Distribution of Y-Coordinate Errors' % cfg.NAME)
    ax.set(xlabel='Normal Distribution M=' + str(errors_y.mean()) + ' SD=' + str(errors_y.std()), ylabel='Frequency')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, str(axis_title[0]).split("'")[-2]))
    plt.show()

    ax = sns.distplot(errors_z)
    axis_title = ax.set(title='Validation %s Distribution of Z-Coordinate Errors' % cfg.NAME)
    ax.set(xlabel='Normal Distribution M=' + str(errors_z.mean()) + ' SD=' + str(errors_z.std()), ylabel='Frequency')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, str(axis_title[0]).split("'")[-2]))
    plt.show()


if __name__ == "__main__":
    run(evaluate)
