#!/bin/bash

echo "Installing Python Libraries"
pip3 install --user --upgrade pip
pip3 install --no-cache-dir --user -r /requirements.txt

