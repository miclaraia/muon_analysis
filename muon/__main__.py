#!/usr/bin/env python
################################################################
# Main entrypoint for the processing algorithm

import muon
from muon import ui

import logging
logger = logging.getLogger(muon.__name__)


def main():
    try:
        ui.run()
    except Exception as e:
        logger.critical(e)
        raise e


if __name__ == "__main__":
    main()
