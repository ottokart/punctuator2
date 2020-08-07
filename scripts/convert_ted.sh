#!/bin/bash
set -e
python convert_ted.py ../data/raw/LREC/dev2012 ../data/postprocessed/ted.dev.txt
python convert_ted.py ../data/raw/LREC/test2011 ../data/postprocessed/ted.test.txt
python convert_ted.py ../data/raw/LREC/test2011asr ../data/postprocessed/ted.test-asr.txt
python convert_ted.py ../data/raw/LREC/train2012 ../data/postprocessed/ted.train.txt
