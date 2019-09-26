#!/bin/bash
set -e
. .env/bin/activate
python setup.py sdist
twine upload dist/*
