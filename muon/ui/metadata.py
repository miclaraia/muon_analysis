
from muon.ui import ui
import muon.swap.muon_metadata as mm
import muon.utils.hdf5 as hdf5
import swap.config

import click
import code

@ui.cli.group()
def meta():
    pass


@meta.command()
@click.argument('file', nargs=1)
def subject_id(file):
    mm.SubjectID.run(file)


@meta.command()
@click.argument('file', nargs=1)
def test_regex(file):
    mm.SubjectID.test_regex(file)

@meta.command()
def regex():
    import code
    r = mm.SubjectID.regex
    p = r.pattern
    print()
    print(p)
    print()

    s = ''
    def test(p=p, s=s):
        r = mm.re.compile(p)
        return r.match(s)
    code.interact(local=locals())

@ui.cli.group()
def hdf():
    pass

@hdf.command()
@click.argument('path', nargs=1)
def test_hdf(path):
    swap.config.logger.init()

    s = hdf5.Subjects()
    subjects = s.subjects_from_files(path)

    code.interact(local=locals())


@hdf.command()
@click.argument('path', nargs=-1)
def pca(path):
    swap.config.logger.init()
    subjects = hdf5.Subjects(path)

    hdf5.Cluster.run(subjects)



