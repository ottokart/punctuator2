# Punctuator

A bidirectional recurrent neural network model with attention mechanism for restoring missing inter-word punctuation in unsegmented text.

The model can be trained in two stages (second stage is optional):

1. First stage is trained on punctuation annotated text. Here the model learns to restore puncutation based on textual features only.
2. Optional second stage can be trained on punctuation *and* pause annotated text. In this stage the model learns to combine pause durations with textual features and adapts to the target domain. If pauses are omitted then only adaptation is performed. Second stage with pause durations can be used for example for restoring punctuation in automatic speech recognition system output.

# How well does it work?

A working demo can be seen here: http://bark.phon.ioc.ee/punctuator

Remember that all the scores given below are on _unsegmented_ text and we did not use prosodic features, so, among other things, the model has to detect sentence boundaries in addition to the boundary type (?QUESTIONMARK, .PERIOD or !EXCLAMATIONMARK) based entirely on textual features. The scores are computed on the test set.

## English TED talks
Training set size: 2.1M words. First stage only. More details can be found in [this paper](http://www.isca-speech.org/archive/Interspeech_2016/pdfs/1517.PDF).
For comparison, our [previous model](https://github.com/ottokart/punctuator) got an overall F1-score of 50.8.

PUNCTUATION      | PRECISION | RECALL    | F-SCORE
--- | --- | --- | ---
,COMMA           | 64.4 | 45.2 | 53.1
?QUESTIONMARK    | 67.5 | 58.7 | 62.8
.PERIOD          | 72.3 | 71.5 | 71.9
_Overall_          | _68.9_ | _58.1_ | _63.1_

## English Europarl v7
Training set size: 40M words. First stage only. First 80% of lines as training set, next 10% as dev set and last 10% as test set.

PUNCTUATION      | PRECISION | RECALL    | F-SCORE
--- | --- | --- | ---
?QUESTIONMARK | 76.9 | 73.5 | 75.2
!EXCLAMATIONMARK | 25.0 | 0.1 | 0.1
,COMMA | 69.2 | 71.8 | 70.5
-DASH | 56.6 | 7.3 | 13.0
:COLON | 56.8 | 25.4 | 35.1
;SEMICOLON | 56.3 | 1.2 | 2.3
.PERIOD | 84.4 | 84.4 | 84.4
_Overall_          | _75.8_      | _73.9_      | _74.8_

# Requirements
* Python
* Numpy
* Theano

# Requirements for data:

* Cleaned text files for training and validation of the first phase model. Each punctuation symbol token must be surrounded by spaces.

  Example:
  ```to be ,COMMA or not to be ,COMMA that is the question .PERIOD```
* *(Optional)* Pause annotated text files for training and validation of the second phase model. These should be cleaned in the same way as the first phase data. Pause durations in seconds should be marked after each word with a special tag `<sil=0.200>`. Punctuation mark, if any, must come after the pause tag.

  Example:
  ```to <sil=0.000> be <sil=0.100> ,COMMA or <sil=0.000> not <sil=0.000> to <sil=0.000> be <sil=0.150> ,COMMA that <sil=0.000> is <sil=0.000> the <sil=0.000> question <sil=1.000> .PERIOD```

  Second phase data can also be without pause annotations to do just target domain adaptation.
  
Make sure that first words of sentences don't have capitalized first letters. This would give the model unfair hints about period locations.

# Configuration
Vocabulary size, punctuation tokens and their mappings, and converted data location can be configured in the header of data.py.
Some model hyperparameters can be configured in the headings of main.py and main2.py. Learning rate and hidden layer size can be passed as arguments.

# Usage

First step is data conversion. Assuming that preprocessed and cleaned *.train.txt, *.dev.txt and *.test.txt files are located in `<data_dir>`, the conversion can be initiated with:

`python data.py <data_dir>`

If you have second stage data as well, then:

`python data.py <data_dir> <second_stage_data_dir>`



The first stage can be trained with:

`python main.py <model_name> <hidden_layer_size> <learning_rate>`

e.g `python main.py <model_name> 256 0.02` works well.



Second stage can be trained with:

`python main2.py <model_name> <hidden_layer_size> <learning_rate> <first_stage_model_path>`



Preprocessed text can be punctuated with e.g:

`cat data.dev.txt | python punctuator.py <model_path> <model_output_path>`

Punctuation tokens in data.dev.txt don't have to be removed - the punctuator.py script ignores them.


Error statistics in this example can be computed with:

`python error_calculator.py data.dev.txt <model_output_path>`


You can play with a trained model with:

`python play_with_model.py <model_path>`


# Citing

The software is described in:

    @inproceedings{tilk2016,
      author    = {Ottokar Tilk and Tanel Alum{\"a}e},
      title     = {Bidirectional Recurrent Neural Network with Attention Mechanism for Punctuation Restoration},
      booktitle = {Interspeech 2016},
      year      = {2016}
    }
