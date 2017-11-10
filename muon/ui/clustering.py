from muon.ui import ui
import muon.utils.clustering as clustering
import swap.config

import click
import code
import pickle


@ui.cli.group()
def pca():
    pass

@pca.command()
@click.argument('path', nargs=1)
def test_pca(path):
    swap.config.logger.init()

    s = clustering.Subjects()
    subjects = s.subjects_from_files(path)

    code.interact(local=locals())


@pca.command()
@click.argument('path', nargs=-1)
def run(path):
    swap.config.logger.init()
    subjects = clustering.Subjects(path)

    clustering.Cluster.run(subjects)

@pca.command()
@click.argument('path', nargs=1)
def load(path):
    pickle.load(path)

    code.interact(local=locals())
