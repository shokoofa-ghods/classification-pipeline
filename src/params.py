"""
Program parameter module
"""

import argparse
import os
import json
from dataclasses import dataclass
from typing import Any
import ast

from utils import parse # pylint: disable=import-error

@dataclass
class Params:
    """
    Parameter data class
    Should contain all information needed to the program
    """
    file_address:str
    data_address:str
    train_info_csv:str
    val_info_csv:str
    test_info_csv:str
    is_multiframe:bool
    verbose:bool
    plot:bool
    tensorboard:bool
    loss_limit:int
    after_n_epochs:int
    epochs:int
    lr:float
    batch_size:int
    base_model:str
    data_mean:list
    data_std:list

    def __str__(self):
        return str(vars(self)).replace(', ', ',\n')

_DEFAULT_PARAMS = {
    'file_address': '',
    'data_address': '',
    'train_info_csv': '',
    'val_info_csv': '',
    'test_info_csv': '',
    'is_multiframe': False,
    'verbose': False,
    'plot': False,
    'tensorboard': False,
    'loss_limit': 4,
    'after_n_epochs' : 5,
    'epochs': 50,
    'lr': 1E-3,
    'batch_size': 32,
    'base_model': '',
    'data_mean' : [0.05, 0.05, 0.05],
    'data_std' : [0.05, 0.05, 0.05]
}

class ParamBuilder:
    """
    Param builder class
    """
    def __init__(self,
                 fill_missing:bool=True,
                 required:list[str] | None=None,
                 defaults:dict[str, Any] | None=None
                 ) -> None:
        self._properties = {}
        self._required = required if required is not None else []
        self._defaults = defaults if defaults is not None else {}
        self._fill_missing = fill_missing


    def build(self) -> Params:
        """
        Create the Params object from the builder
        Raises:
            ValueError if a required field has not been set
        Returns:
            The Params object built with default values filled
        """
        for field in self._required:
            if field not in self._properties:
                raise ValueError(f'Required field "{field}" has not been set')

        return Params(file_address=self['file_address'],
                      data_address=self['data_address'],
                      train_info_csv=self['train'],
                      val_info_csv=self['eval'],
                      test_info_csv=self['test'],
                      is_multiframe=self['multiframe'],
                      verbose=self['verbose'],
                      plot=self['plot'],
                      tensorboard=self['tensorboard'],
                      loss_limit=self['loss_limit'],
                      after_n_epochs = self['after_n_epochs'],
                      epochs=self['epochs'],
                      lr=self['lr'],
                      batch_size=self['batch_size'],
                      base_model=self['base_model'],
                      data_mean=ast.literal_eval(self['data_mean']),
                      data_std=ast.literal_eval(self['data_std']),
                    )

    def set(self, override:bool=True, **kwargs):
        """
        Set one or more properties in the builder
        example: builder.set(origin='/home', lr=0.1)
        """
        for k, v in kwargs.items():
            if v is None:
                continue
            if k in self._properties and not override:
                continue
            self._properties[k] = v

    def __getitem__(self, key):
        if key in self._properties:
            return self._properties[key]
        if key in self._defaults:
            return self._defaults[key]
        if self._fill_missing and key in _DEFAULT_PARAMS:
            print(f'key "{key}" not found. Using default "{_DEFAULT_PARAMS[key]}"')
            return _DEFAULT_PARAMS[key]
        raise KeyError(f'key "{key}" not found and has no default')

_parser = argparse.ArgumentParser()
_parser.add_argument('-c', '--config',
                    required=False,
                    dest='config',
                    help='Include to use specified config file')
_parser.add_argument('-v', '--var',
                    required=False,
                    action='append',
                    default=[],
                    dest='vars',
                    help='Set program vars "-v <var1>=<value1> -v <var2>=<value2>"')

def get_params(argv:list[str],
               required:list[str] | None=None,
               defaults:dict[str, Any] | None=None
               ) -> Params:
    """
    Get the parameters from the argv
    if a config file is specified it will be used first and command
    line arguments will be used as defaults (it is recommended that all
    parameters be provided by the config file if using one as flags will
    be interpreted as False if not present which may cause confusion for
    required parameters (i.e. multiframe)).

    Args:
        argv (list[str]): The program arguments to check through
    Raises:
        ValueError: a required parameter is missing or the config file does not exist
    Returns:
        A Params object with all the parameters
    """
    args = _parser.parse_args(argv)
    builder = ParamBuilder(required=required, defaults=defaults)
    if args.config is not None:
        # Verify config file exists
        if not os.path.exists(args.config):
            err_msg = f'ERROR: config file "{args.config}" does not exist'
            raise ValueError(err_msg)
        # Read the json data
        with open(args.config, encoding='utf-8') as config_file:
            config = json.load(config_file)
        # Add all parameters found in the config to the builder
        builder.set(**config)
    # Add all command line args to builder
    for arg in args.vars:
        k, v = arg.split('=')[:2]
        builder.set(override=False, **{k:parse(v)[0]})

    return builder.build()
