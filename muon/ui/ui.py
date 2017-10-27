
import argparse
import logging
logger = logging.getLogger(__name__)


class BaseUI:
    _interfaces = []
    command = 'ROOT'

    def __init__(self):
        self.interfaces = {}
        self.parser = argparse.ArgumentParser()

        self.options(self.parser)

        print(self.__class__, self._interfaces)
        if len(self._interfaces) > 0:
            sub_parser = self.parser.add_subparsers(dest=self.command)

            for interface in self._interfaces:
                # if type(interface) is Interface:
                    # continue
                self.register(sub_parser, interface())

    def register(self, sub_parser, interface):
        """
        Register an interface with the UI

        Parameters
        ----------
        interface : ui.Interface
            Interface to be added
        """
        command = interface.command

        parser = sub_parser.add_parser(command)
        parser.set_defaults(dest=command)

        interface.options(parser)
        self.interfaces[command] = interface

    def run(self):
        """
        Called after interfaces have registered to parse arguments and
        execute operations
        """
        args = self.parser.parse_args()
        logger.debug(self.interfaces)
        logger.debug(args)
        self._call(args)

    def options(self, parser):
        pass

    def call(self, args):
        pass

    def _call(self, args):
        self.call(args)
        if self.command in args:
            command = vars(args)[self.command]
            self.interfaces[command]._call(args)


class UI(BaseUI):
    """
    Main interface endpoint, manages interaction with argparse. Interfaces
    register with a UI instance, and the UI instance chooses the right
    Interface to pass args to.
    """

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
        pass


def class_register(cls):
    UI._interfaces.append(cls)
    return cls


class MetaInterface(type):
    def __new__(mcs, clsname, bases, attrs):
        cls = super().__new__(mcs, clsname, bases, attrs)
        cls.__bases__[0]._interfaces.append(cls)
        cls._interfaces = []

        return cls


# class Interface(UI, metaclass=MetaInterface):
    # """
    # Interface that defines a set of options and operations.
    # Designed to be subclassed and overriden
    # """

    # def __init__(self):
        # """
        # Initialize this interface and register it with the UI.

        # Parameters
        # ----------
        # ui : ui.UI
            # UI to register with
        # """
        # pass

    # @property
    # def command(self):
        # """
        # Command used to select parser.

        # For example, this would return 'swap' for SWAPInterface
        # and 'roc' for RocInterface
        # """
        # pass

    # def options(self, parser):
        # """
        # Add options to the parser
        # """
        # pass

    # def call(self, args):
        # """
        # Define what to do if this interface's command was passed
        # """
        # pass

    ##############################################################



