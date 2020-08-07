This example produces the preprocessed Europarl English corpus that can be then used for training a model.

Requires nltk

Usage example:
`./run.sh`

`cd ..`

`python data.py ./example/out`

`python main.py ep 256 0.02`

`python play_with_model.py Model_ep_h256_lr0.02.pcl`

The input text to play_with_model.py should be similar to the contents of the preprocessed files in ./example/out (i.e. lowercased, numeric tokens replaced with <NUM>), but should not contain punctuation tokens.

Training time on this dataset with a Nvidia Tesla K20 GPU was about 15 hours (~3500 samples per second)