import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import click

from muon.dissolving.utils import Metrics


@click.group(invoke_without_command=True)
@click.argument('metrics')
@click.argument('splits_file')
def main(metrics, splits_file):
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
        x_train = splits['train'][0]
        x_train_dev = splits['train_dev'][0]
    fname = metrics
    with open(fname, 'rb') as f:
        metrics = pickle.load(f)

        for i in range(len(metrics.metrics)):
            if metrics.metrics[i]['iteration'] == metrics.redec_mark:
                break

    batches = np.ceil((x_train.shape[0]+x_train_dev.shape[0])/256) \
        .astype(np.int)
    # print(i, batches, metrics.redec_mark, metrics.metrics[i])
    # print(metrics.metrics[80-1])
    for i in range(i, len(metrics.metrics)):
        metrics.metrics[i]['iteration'] = (metrics.metrics[i]['iteration']-80)/batches+80

    print(metrics.print_ite(metrics.last_ite()))
    # fig = plt.figure(figsize=(15, 8))
    # metrics.plot(fig)
    # plt.show()

    with open(fname, 'wb') as file:
        pickle.dump(metrics, file)


if __name__ == '__main__':
    main()

