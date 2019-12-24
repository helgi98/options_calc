#!/bin/bash
# installs virtualenv
pip install virtualenv

# set ups virtual environment
virtualenv  --no-site-packages venv

source venv/Scripts/activate

# updates pip
pip install --upgrade setuptools

# installs pyqt5
pip install pyqt5

# installs matpotlib
pip install -f  http://garr.dl.sourceforge.net/project/matplotlib/matplotlib/matplotlib-1.0.1/matplotlib-1.0.1.tar.gz  matplotlib

# installs numpy
pip install numpy

# installs scipy
pip install scipy

deactivate

printf '\n\nPress enter to finish'
read -s -n 1