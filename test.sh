#!/bin/bash
# Runs all tests.
set -e
./pep8.sh
export TESTNAME=; tox
