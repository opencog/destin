#!/bin/sh
cd "$(dirname "$0")"
# remove compiled python files 
# so a missing python file doesn't pretend it's there.
rm *.pyc
python test.py

