from muon.ui import ui
from muon.utils.clustering import Cluster
from muon.utils.subjects import Subjects
import muon.scripts.clustering as scripts
import swap.config

import click
import code
import pickle


@ui.cli.group()
def pca():
    pass

@pca.command()
@click.argument('path', nargs=-1)
def run(path):
    swap.config.logger.init()
    subjects = Subjects.from_data(path)

    cluster = Cluster.create(subjects)
    cluster.plot()
    code.interact(local=locals())

@pca.command()
@click.argument('path', nargs=1)
def load(path):
    cluster = pickle.load(open(path, 'rb'))

    code.interact(local=locals())

@pca.command()
@click.argument('path', nargs=1)
def visualize(path):
    swap.config.logger.init()
    subjects = Subjects.from_data(path)
    cluster = Cluster.create(subjects)
    cluster.visualize()


@pca.command()
@click.argument('path', nargs=-1)
@click.argument('script', nargs=1)
def cluster(path, script):
    if script == '1':
        scripts.regions(path)
    

