#!/bin/bash
set -e

cd "$(dirname "$0")"

CACHE_DIR=/tmp/pip
REL_DIR=./

# Remove existing virtualenv if it exists.
[ -d $REL_DIR.env ] && rm -Rf $REL_DIR.env

# Create virtual environment with Python 3.7 (requires python3-venv package on Ubuntu)
python3.7 -m venv $REL_DIR.env
. $REL_DIR.env/bin/activate
pip install -U pip
pip install -U setuptools
pip install pypandoc

pip install --cache-dir $CACHE_DIR -r requirements.txt -r requirements-test.txt
