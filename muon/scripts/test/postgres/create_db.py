import logging

from muon.database.database import Database

logger = logging.getLogger(__name__)


database = Database('localhost', 'muon_data')
database.create_db()
