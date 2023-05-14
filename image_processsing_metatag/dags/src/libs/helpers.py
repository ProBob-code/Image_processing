import toml
from functools import reduce
from operator import getitem
from pathlib import Path
import string
import random
import re


def checkKeyExists(dictionary, key):
    try:
        if key in dictionary.keys():
            return True
        else:
            return False
    except Exception as e:
        return False


parsed_toml_cache = {}

def load_parsed_toml(filename):
    project_directory = str(Path(__file__).parent.parent)

    if filename in parsed_toml_cache:
        return parsed_toml_cache[filename]
    # with open(filename, 'r') as f:
    #     parsed_toml = toml.load(project_directory + f)

    parsed_toml = toml.load(project_directory + filename)

    parsed_toml_cache[filename] = parsed_toml
    return parsed_toml

def getConfigData(key, index=None):
    try:
        # config = toml.load(project_directory + '/config.toml')
        config = load_parsed_toml('/config.toml')
        path = key.split('.')
        response = reduce(getitem, path, config)
        if index is None:
            return response
        else:
            return response[index]
    except Exception as e:
        return None


def arrayIntersect(array1, array2):

    try:

        array3 = [value for value in array1 if value in array2]
        return array3

    except Exception as e:
        
        return []

def randomStringGenerator(size=10, chars=string.ascii_uppercase + string.digits):

    return ''.join(random.choice(chars) for _ in range(size))


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
        
    # try:
    #     return data.encode('utf-8')
    # except UnicodeDecodeError:
    #     data = data.decode('iso-8859-1')
    #     for _hex, _char in LATIN_1_CHARS:
    #         data = data.replace(_hex, _char)
    #     return data.encode('utf8')

    try:
        data = data.decode('iso-8859-1')
        for _hex, _char in LATIN_1_CHARS:
            data = data.replace(_hex, _char)
        return data
    except Exception as e:
        return data


def isValidEmail(email):
    try:
        
        regex   = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email   = email.strip()

        if(re.fullmatch(regex, email)):
            return True
        else:
            return False

    except Exception as e:
        return False


def isValidMobile(mobile):
    try:
        
        Pattern = re.compile("[0-9][0-9]{9}")
        return Pattern.match(mobile.strip())

    except Exception as e:
        return False

def validateMobileNumbers(mobile):
    
    try:
        
        valid_mobile_arr    = []
        mobile_arr          = mobile.split(',')
        pattern             = re.compile("[0-9][0-9]{9}")
        
        if len(mobile_arr) > 0:

            for mobile_str in mobile_arr:
                
                numeric_filter = filter(str.isdigit, mobile_str)
                mobile_number = "".join(numeric_filter)
                
                if mobile_number[0:3] == '+91':
                    mobile_number   = mobile_number[3:]
                elif mobile_number[0:2] == '91' and len(mobile_number) > 10:
                    mobile_number   = mobile_number[2:]
                elif mobile_number[0:1] == '0':
                    mobile_number   = mobile_number[1:]
                    
                if (pattern.match(mobile_number.strip())):
                   valid_mobile_arr.append(mobile_number)

        return valid_mobile_arr

    except Exception as e:

        return mobile.split(',')


def validateEmails(email):
    
    try:
        
        valid_email_arr    = []
        email_arr          = email.split(',')
        regex              = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        if len(email_arr) > 0:

            for email_str in email_arr:
                email_str   = email_str.strip()
                if(re.fullmatch(regex, email_str)):
                   valid_email_arr.append(email_str)

        return valid_email_arr

    except Exception as e:

        return email.split(',')

def addSlashes(string):

    try:

        if string != None and string.strip() != '':

            string  = string.strip()
            string  = string.replace("'", "\'")
            string  = string.replace('"', '\"')
        
        return string

    except Exception as e:

        return string

def addSlashesWithDoubleSlash(string):

    try:

        if string != None and string.strip() != '':

            string  = string.strip()
            string  = string.replace("'", "\\'")
            string  = string.replace('"', '\\"')
        
        return string

    except Exception as e:

        return string

def join_coma_separated_strings (old, new):
    try:

        updated_string = ", ".join(set(old.split(",") + new.split(",")))

        return updated_string

    except Exception as e:

        return e
