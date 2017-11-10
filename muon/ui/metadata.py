from muon.ui import ui
import muon.swap.muon_metadata as mm

import click

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
