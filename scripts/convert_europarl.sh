#!/bin/bash
set -e

HOME_DIR="$(dirname "$(readlink -f "$0")")"
DEST=../raw/training-monolingual-europarl.tgz
WORKING_DIR=../raw/training-monolingual-europarl
POSTPROCESSED_DIR=../data/postprocessed

if [ -f "$DEST" ]; then
    echo "$DEST exists. Skipping download."
else
    echo "$DEST does not exist. Downloading..."
    wget -qO- http://hltshare.fbk.eu/IWSLT2012/training-monolingual-europarl.tgz --output-document $DEST
    tar xvz $DEST
fi

echo "Step 1/3"
mkdir -p $WORKING_DIR/out
grep -v " '[^ ]" $WORKING_DIR/europarl-v7.en | \
grep -v \'\ s\   | \
grep -v \'\ ll\  | \
grep -v \'\ ve\  | \
grep -v \'\ m\   > $WORKING_DIR/out/step1.txt

echo "Step 2/3"
python convert_europarl.py $WORKING_DIR/out/step1.txt $WORKING_DIR/out/step2.txt

echo "Step 3/3"
head -n -400000 $WORKING_DIR/out/step2.txt > $POSTPROCESSED_DIR/europarl.train.txt
tail -n 400000 $WORKING_DIR/out/step2.txt > $WORKING_DIR/out/step3.txt
head -n -200000 $WORKING_DIR/out/step3.txt > $POSTPROCESSED_DIR/europarl.dev.txt
tail -n 200000 $WORKING_DIR/out/step3.txt > $POSTPROCESSED_DIR/europarl.test.txt

#echo "Cleaning up..."
#rm -f step1.txt step2.txt step3.txt
echo "Preprocessing done. Now you can give the produced $POSTPROCESSED_DIR dir as <data_dir> argument to data.py script for conversion and continue as described in the main README.md"
