
from muon.ui import ui
import muon.utils.camera
import code

import click

@ui.cli.group()
def camera():
    pass


@camera.command()
def print():
    camera = muon.utils.camera.Camera()
    coords = camera.map_coordinates()

    from pprint import pprint
    pprint(coords)

@camera.command()
def rotate():
    cr = muon.utils.camera.CameraRotate(None)
    d = cr.create_map()

    code.interact(local=locals())
