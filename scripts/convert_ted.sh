#!/bin/bash
set -e
python convert_ted.py ../raw/LREC/dev2012 ../raw/postprocessed/ted.dev.txt
python convert_ted.py ../raw/LREC/test2011 ../raw/postprocessed/ted.test.txt
python convert_ted.py ../raw/LREC/test2011asr ../raw/postprocessed/ted-asr.test.txt
python convert_ted.py ../raw/LREC/train2012 ../raw/postprocessed/ted.train.txt
