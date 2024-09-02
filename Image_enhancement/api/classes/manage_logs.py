import psutil
import time
import logging
import json
# from helper import getConfigData
import os
from datetime import datetime
from flask import has_request_context, request
import socket
from datetime import date
import logging
import toml
from functools import reduce
from operator import getitem
import traceback
from logging import handlers

def getConfigData(key, index=None):
    try:
        config = toml.load('config.toml')
        path = key.split('.')
        response = reduce(getitem, path, config)
        if index is None:
            print("NFS_path: ", response)
            return response
        else:
            return response[index]
    except Exception as e:
        print('error_traceback')
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        print("".join(tb_str[-3:]))
        return None
    
#IPAddr = socket.gethostbyname(hostname)


# Create a custom formatter to include timestamp
class TimedFormatter(logging.Formatter):
    def format(self, record):
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
            input_req = request.form.to_dict()
            record.in_request = json.dumps(input_req, indent = 4)
            hostname = socket.gethostname()
            record.remote_ip = socket.gethostbyname(hostname)
        else:
            record.url = None
            record.remote_addr = None
            record.in_request = None
            record.remote_ip = None
        return super().format(record)
    
    def formatTime(self, record, datefmt=None):
        # Use current time with microseconds for higher precision
        now = datetime.now()
        return now.isoformat(sep=' ', timespec='microseconds')
    
# NFS_path = getConfigData('NFS_path.path')
today = date.today()
# Format the date string in YYYY-MM-DD format
formatted_date = today.strftime("%Y-%m-%d")


# Configure logging (optional, adjust as needed)
handler = logging.FileHandler('server_monitor.log', mode='a')  # Use 'a' for appending
# handler = logging.FileHandler('server_monitor.log', mode='a')  # Use 'a' for appending

# # Create the formatter with timestamp
formatter = TimedFormatter("[%(asctime)s] IP:%(remote_addr)s requested_url:%(url)s\n%(levelname)s in %(module)s:\nREQUEST:\n%(in_request)s\nMESSAGE:\n%(message)s\nIP_SERVER:\n%(remote_ip)s")

handler.setFormatter(formatter)


logging.basicConfig(level=logging.INFO, handlers=[handler])
# logging.basicConfig(filename=NFS_path+'/server_monitor.log', level=logging.INFO)

class Manage_Logs():
    def __init__(self):
        pass
        
    def input_request(local_path, product_url, api_name, e="image corrupted or deleted"):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")  # Format the time
        logging.info(f"Current Time: {current_time}\napi_name: {api_name}\nError (try-except):\n{e}\nlocal_path: {local_path}\nproduct_url: {product_url}\n\n")

