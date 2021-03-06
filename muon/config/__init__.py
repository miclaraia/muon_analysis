
import os
import yaml
import logging

logger = logging.getLogger(__name__)


class Config:
    _instance = None

    def __init__(self, path=None):
        self.config = {}

        paths = [
            path,
            os.environ.get('MUON_CONFIG'),
            os.path.join(os.environ.get('HOME'), '.muon_config.yml'),
            os.path.join(os.path.dirname(__file__), 'default_config.yml')
        ]

        self.path = None
        for path in paths:
            if path is not None and os.path.isfile(path):
                self.path = path
                break
        print(self.path)

        if self.path is None:
            raise FileNotFoundError('No valid config file found')

        with open(self.path, 'r') as f:
            self.config = yaml.load(f)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls.new()
        return cls._instance

    @classmethod
    def new(cls, *args, **kwargs):
        self = cls(*args, **kwargs)
        cls._instance = self
        return self

    @property
    def classification(self):
        return Classification(self.config)

    @property
    def panoptes(self):
        return Panoptes(self.config)

    @property
    def plotting(self):
        return Plotting(self.config)

    @property
    def storage(self):
        return Storage(self.config)

    @property
    def database(self):
        return Database(self.config)

    @property
    def loglevel(self):
        return self.config['general']['loglevel']


class SubConfig:

    def __init__(self, config):
        self.config = config


class Plotting(SubConfig):

    @property
    def _config(self):
        return self.config['plotting']

    @property
    def cmap(self):
        return self._config['cmap']

    @property
    def dpi(self):
        return self._config['dpi']


class Panoptes(SubConfig):

    @property
    def _config(self):
        return self.config['panoptes']

    @property
    def project_id(self):
        project_id = os.environ.get('MUON_PROJECT')
        return project_id or self._config['project_id']


class Storage(SubConfig):

    @property
    def _config(self):
        return self.config['storage']

    @staticmethod
    def _pre(path):
        if 'MUOND' in path:
            path = path.replace('MUOND', os.environ.get('MUOND'))
        return path

    @property
    def database(self):
        return self._pre(self._config['database'])

    @property
    def images(self):
        return self._pre(self._config['images'])


class Database(SubConfig):

    @property
    def _config(self):
        return self.config['database']

    @property
    def host(self):
        return self._config['host']

    @property
    def dbname(self):
        return self._config['dbname']

    @property
    def username(self):
        return self._config['username']

    @property
    def password(self):
        return self._config['password']


class Classification(SubConfig):

    @property
    def _config(self):
        return self.config['classification']

    @property
    def tool_name(self):
        return self._config['tool_name']

    @property
    def task_A(self):
        return self._config['task_A']

    @property
    def task_B(self):
        return self._config['task_B']

    @property
    def launch_date(self):
        return self._config['launch_date']

    @property
    def time_format(self):
        return self._config['time_format']

    @property
    def image_groups(self):
        return self._config['image_groups']

    @property
    def task_map(self):
        task_map = self._config['task_map']
        return {k: [i for i in task_map[k]] for k in task_map}

