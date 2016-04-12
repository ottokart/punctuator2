# Punctuator

A bidirectional recurrent neural network model with attention mechanism for restoring missing punctuation in text.

A working demo can be seen here: http://bark.phon.ioc.ee/punctuator

Model can be trained in two stages (second stage is optional):

1. First stage is trained on punctuation annotated text. Here the model learns to restore puncutation based on textual features only.
2. Optional second stage can be trained on punctuation *and* pause annotated text. In this stage the model learns to combine pause durations with textual features and adapts to the target domain. If pauses are omitted then only adaptation is performed. Second stage with pause durations can be used for example for restoring punctuation in automatic speech recognition system output.

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

`python main.py <model_name> <hidden_layer_size> <learning_rate> <first_stage_model_path>`



Preprocessed text can be punctuated with e.g:

`cat data.dev.txt | python punctuator.py <model_path> <model_output_path>`

Punctuation tokens in data.dev.txt don't have to be removed - the punctuator.py script ignores them.


Error statistics in this example can be computed with:

`python error_calculator.py data.dev.txt <model_output_path>`


You can play with a trained model with:

`python play_with_model.py <model_path>`