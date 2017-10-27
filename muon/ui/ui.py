
import argparse
import logging
logger = logging.getLogger(__name__)


class UI:
    """
    Main interface endpoint, manages interaction with argparse. Interfaces
    register with a UI instance, and the UI instance chooses the right
    Interface to pass args to.
    """

    _interfaces = []

    def __init__(self):
        self.interfaces = {}
        self.parser = argparse.ArgumentParser()
        self.sparsers = self.parser.add_subparsers()

        self.dir = None

        self.options(self.parser)

        for interface in self._interfaces:
            if type(interface) is Interface:
                continue
            self.register(interface())

    def run(self):
        """
        Called after interfaces have registered to parse arguments and
        execute operations
        """
        print(self._interfaces)
        args = self.parser.parse_args()
        logger.debug(args)
        self.call(args)

    def options(self, parser):
        """
        Adds arguments to the parser

        Parameters
        ----------
        parser : argparse.ArgumentParser
            parser to add args to
        """
        pass

    def call(self, args):
        """
            Called when executing args

            Parameters
            ----------
            args : argparse.Namespace
        """
        if 'func' in args:
            args.func(args)

    def register(self, interface):
        """
        Register an interface with the UI

        Parameters
        ----------
        interface : ui.Interface
            Interface to be added
        """
        command = interface.command

        sparser = self.sparsers.add_parser(command)
        sparser.set_defaults(func=interface.call)

        interface.options(sparser)
        self.interfaces[command] = interface


def class_register(cls):
    UI._interfaces.append(cls)
    return cls

class MetaInterface(type):
    def __new__(cls, clsname, bases, attrs):
        cls = super().__new__(cls, clsname, bases, attrs)
        print(str(cls.__class__.__name__))
        if str(cls) != '<class \'muon.ui.ui.Interface\'>':
            UI._interfaces.append(cls)
        return cls


class Interface(metaclass=MetaInterface):
    """
    Interface that defines a set of options and operations.
    Designed to be subclassed and overriden
    """

    def __init__(self):
        """
        Initialize this interface and register it with the UI.

        Parameters
        ----------
        ui : ui.UI
            UI to register with
        """
        # self.ui = ui
        # ui.add(self)
        self.init()

    def init(self):
        """
        Method called on init, after having registered with ui
        """
        pass

    @property
    def command(self):
        """
        Command used to select parser.

        For example, this would return 'swap' for SWAPInterface
        and 'roc' for RocInterface
        """
        pass

    def options(self, parser):
        """
        Add options to the parser
        """
        pass

    def call(self, args):
        """
        Define what to do if this interface's command was passed
        """
        pass

    ###############################################################



