import os
import json
from glob import glob
import ntpath
from typing import List
import re
from argparse import Namespace

def load_config(config_file_path):
    """
    Load the configuration file in JSON format and return it as a dictionary.

    Args:
        config_file_path (str): Path to the configuration file.

    Returns:
        dict: Configuration values from the file.
    """
    if not os.path.exists(config_file_path):
        return False
    
    with open(config_file_path, 'r') as file:
        config_data = json.load(file)    
    return config_data

def get_token(path, complete=True):
    if complete == False:
        base_path = '/projects/klybarge/muhammad_research/'
        path = os.path.join(base_path,path)
    with open(path, 'r') as file:
        # Read the content of the file
        TOKEN = file.read().strip()  # Stripping any unnecessary whitespace/newlines
    return TOKEN

def update_arguments_with_config(parser, config_data, args):
    """
    Update the argument parser values based on the config file and parsed arguments.

    Args:
        parser (ArgumentParser): The argument parser object.
        config_data (dict): Dictionary of configuration values from the config file.
        args (Namespace): Parsed arguments from the command line.

    Returns:
        Namespace: Updated argument values with proper priority.
    """
    if not config_data:
        return parser.parse_args()
    # Convert parsed arguments to a dictionary
    parsed_args = vars(args)

    # Update argument parser defaults based on config file if command line arguments are not provided
    for key, value in config_data.items():
        if key not in parsed_args or parsed_args[key] is None:
            parser.set_defaults(**{key: value})

    # Re-parse the arguments with updated defaults
    final_args = parser.parse_args()
    
    return final_args

def update_args(args, config_data):
    """
    Update the argument values based on the config file and parsed arguments.

    Args:
        config_data (dict): Dictionary of configuration values from the config file.
        args (Namespace): Parsed arguments from the command line.

    Returns:
        Namespace: Updated argument values with proper priority.
    """
    # Convert parsed arguments (args) to a dictionary
    parsed_args = vars(args)
    
    # Iterate over config_data and add/update arguments
    for key, value in config_data.items():
        if key not in parsed_args or parsed_args[key] is None:
            # Update the namespace with new or overridden values
            setattr(args, key, value)

    return args