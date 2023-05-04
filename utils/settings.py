"""Age Mapper Settings Loader File

Description:

    This file contains several functions required for loading the settings of BrainMapper.

Usage:

    To use the modules, import the packages and instantiate any module/block class as you wish:

        from settings import Settings

"""

import ast
import configparser
from collections.abc import Mapping


class Settings(Mapping):
    """
    Settings Loader Class

    This class reads a settings file, and loads the data as dictionaries.
    It makes use of configparser, which implements a basic configuration language which provides a structure similar to what's found in Microsoft Windows INI files.
    """

    def __init__(self, 
                 settings_file: str = 'settings.ini'
                 ) -> None:
        """ 
        Settings Loader Class Initialiser
        
        Parameters:
        -----------
        settings_file : str
            Settings file name
        
        Returns:
        --------
        item: str, int or float
            Different str, int and float values corresponding to the various settings dictionary keys
        int:
            Lenght of the various dictionaries
        list:
            A list of a given settings dictionary's (key, value) tuple pairs.

        """

        configurator = configparser.ConfigParser()
        if not configurator.read(settings_file):
            configurator.read('functionmapper/'+settings_file)
        else:
            configurator.read(settings_file)
        self.settings_dictionary = _parse_values(configurator)

    def __getitem__(self, 
                    key: str) -> str:
        return self.settings_dictionary[key]

    def __len__(self):
        return len(self.settings_dictionary)

    def __iter__(self):
        return self.settings_dictionary.items()


def _parse_values(configurator: configparser.ConfigParser) -> dict:
    """ Configurator file reader

    This function reads the various sections in a configurator file.
    Then, for every section, it reads the key-value pairs.
    Using the key-value pairs, it constructs a nested dictionary for each section key.

    Parameters:
    -----------
    configurator : class
        ConfigParser object class

    Returns:
    --------
    settings_dictionary : dict
        Dictionary containing all settings for different sections

    """

    settings_dictionary = {}
    for section in configurator.sections():
        settings_dictionary[section] = {}
        for key, value in configurator[section].items():
            # Safely evaluate an expression node or a Unicode or Latin-1 encoded string containing a Python expression
            settings_dictionary[section][key] = ast.literal_eval(value)

    return settings_dictionary
