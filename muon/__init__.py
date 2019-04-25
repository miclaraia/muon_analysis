
import logging
import muon.config

logging.basicConfig(level=muon.config.Config.instance().loglevel)
