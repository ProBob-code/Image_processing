import toml
from functools import reduce
from operator import getitem
from pathlib import Path
import re
import traceback

def checkKeyExists(dictionary, key):
    try:
        if key in dictionary.keys():
            return True
        else:
            return False
    except Exception as e:
        return False

parsed_toml_cache = {}

# def load_parsed_toml(filename):
#     project_directory = str(Path(__file__).parent.parent)

#     if filename in parsed_toml_cache:
#         return parsed_toml_cache[filename]
#     # with open(filename, 'r') as f:
#     #     parsed_toml = toml.load(project_directory + f)

#     parsed_toml = toml.load(project_directory + filename)

#     parsed_toml_cache[filename] = parsed_toml
#     return parsed_toml

def getConfigData(key, index=None):
    try:
        config = toml.load('config.toml')
        path = key.split('.')
        response = reduce(getitem, path, config)
        if index is None:
            return response
        else:
            return response[index]
    except Exception as e:
        print('error_traceback')
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        print("".join(tb_str[-3:]))
        return None

# def getConfigData1(key, index=None):
#     try:
#         # config = toml.load(project_directory + '/config.toml')
#         config = load_parsed_toml('config.toml')
#         path = key.split('.')
#         response = reduce(getitem, path, config)
#         if index is None:
#             return response
#         else:
#             return response[index]
#     except Exception as e:
#         return None

def clean_latin1(data):
        
    LATIN_1_CHARS = (
        ("\xe2\x80\x99", "'"),
        ("\xc3\xa9", "e"),
        ("\xe2\x80\x90", "-"),
        ("\xe2\x80\x91", "-"),
        ("\xe2\x80\x92", "-"),
        ("\xe2\x80\x93", "-"),
        ("\xe2\x80\x94", "-"),
        ("\xe2\x80\x94", "-"),
        ("\xe2\x80\x98", "'"),
        ("\xe2\x80\x9b", "'"),
        ("\xe2\x80\x9c", '"'),
        ("\xe2\x80\x9c", '"'),
        ("\xe2\x80\x9d", '"'),
        ("\xe2\x80\x9e", '"'),
        ("\xe2\x80\x9f", '"'),
        ("\xe2\x80\xa6", '...'),
        ("\xe2\x80\xb2", "'"),
        ("\xe2\x80\xb3", "'"),
        ("\xe2\x80\xb4", "'"),
        ("\xe2\x80\xb5", "'"),
        ("\xe2\x80\xb6", "'"),
        ("\xe2\x80\xb7", "'"),
        ("\xe2\x81\xba", "+"),
        ("\xe2\x81\xbb", "-"),
        ("\xe2\x81\xbc", "="),
        ("\xe2\x81\xbd", "("),
        ("\xe2\x81\xbe", ")")
    )

    try:
        data = data.decode('iso-8859-1')
        for _hex, _char in LATIN_1_CHARS:
            data = data.replace(_hex, _char)
        return data
    except Exception as e:
        return data

def addSlashes(string):

    try:

        if string != None and string.strip() != '':

            string  = string.strip()
            string  = string.replace("'", "\'")
            string  = string.replace('"', '\"')
        
        return string

    except Exception as e:

        return string

def stripSlashes(string):
    try:

        if string != None and string.strip() != '':

            string = re.sub(r"\\(n|r)", "\n", string)
            string = re.sub(r"\\", "", string)
            
        return string

    except Exception as e:

        return string