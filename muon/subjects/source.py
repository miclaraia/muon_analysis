import hashlib
from datetime import datetime
import dateutil.parser
import os


from muon.database.utils import StorageObject, StorageAttribute, \
    StoredAttribute


class Source(StorageObject):

    hash = StorageAttribute('hash')
    updated = StorageAttribute('updated')
    source_type = StorageAttribute('source_type')

    def __init__(self, source_id, database, attrs=None, online=False):
        # fname, hash=None, updated=None):
        # if type(updated) is str:
            # updated = dateutil.parser.parse(updated)
        # fname = os.path.abspath(fname)

        super().__init__(database, online)

        self.source_id = source_id

        if attrs is None:
            with self.conn as conn:
                attrs = database.Source.get_source(conn, source_id)

        storage = [
            StoredAttribute('hash', attrs['hash']),
            StoredAttribute('updated', attrs['updated']),
            StoredAttribute('source_type', attrs['source_type']),
        ]
        self.storage = {s.name: s for s in storage}

    @classmethod
    def new(cls, source_id, database, location):
        if 'SIM' in source_id:
            source_type = 1
        else:
            source_type = 0

        hash_ = cls._get_hash(location, source_id)
        attrs = {'hash': hash_, 'updated': datetime.now(),
                 'source_type': source_type}
        source = cls(source_id, database, attrs)

        with database.conn as conn:
            database.Source.add_source(conn, source)
            conn.commit()

        return source

    def save(self):
        updates = {}
        for k, v in self.storage.items():
            if v.has_changed:
                if k == 'hash':
                    updates['hash'] = v.value
                    updates['updated'] = datetime.now()
                else:
                    updates[k] = v.value
                v.has_changed = False

        if updates:
            with self.conn as conn:
                self.database.Source \
                    .update_source(conn, self.source_id, updates)
                conn.commit()

    def update_hash(self, location):
        self.hash = self._get_hash(location, self.source_id)
        self.updated = datetime.now()

    @classmethod
    def _get_hash(cls, location, source_id):
        md5 = hashlib.md5()
        with open(os.path.join(location, source_id), 'rb') as f:
            buf = f.read(128)
            while buf:
                md5.update(buf)
                buf = f.read(128)
        return md5.hexdigest()

    def compare(self, location):
        return self._get_hash(location, self.source_id) == self.hash

    def __str__(self):
        source_type = {1: 'sim', 0: 'real'}[self.source_type]
        return '{} {} {} {}'.format(
            self.source_id, source_type, self.hash, self.updated)

    def __repr__(self):
        return str(self)


    # @property
    # def hash(self):
        # if self._hash is None:
            # self._hash = self._get_hash()
            # self.updated = datetime.now()

        # return self._hash

    # def _get_hash(self):
        # md5 = hashlib.md5()
        # with open(self.fname, 'rb') as f:
            # buf = f.read(128)
            # while buf:
                # md5.update(buf)
                # buf = f.read(128)
        # return md5.hexdigest()


