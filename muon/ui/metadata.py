
from muon.ui import ui
import muon.swap.muon_metadata as mm

import click

@ui.cli.group()
def meta():
    pass


@meta.command()
@click.argument('config', nargs=1)
@click.argument('file', nargs=1)
def subject_id(config, file):
    mm.SubjectID.run(config, file)


@meta.command()
@click.argument('config', nargs=1)
@click.argument('file', nargs=1)
def test_regex(config, file):
    mm.SubjectID.test_regex(config, file)

@meta.command()
def regex():
    import code
    r = mm.SubjectID.regex
    p = r.pattern
    print()
    print(p)
    print()

    s = ''
    def test(p):
        r = mm.re.compile(p)
        return r.match(s)
    import code
    code.interact(local=locals())

