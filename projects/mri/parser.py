"""
Requirements to use a custom parser with the general codebase: 
- Create a class that defines self.parser with project-specific args, and pass in this class to parse_to_config
    @args: 
        None
    @rets:
        None; define self.parser within custom_parser class
"""

import argparse

class custom_parser(object):
    """
    Example custom parser
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser("")
        self.parser.add_argument("--custom_arg_1", type=bool, default=True, help='Example of a project-specific argument')
        self.parser.add_argument("--custom_arg_2", type=str, default='Example', help='Another example of a project-specific argument')


    