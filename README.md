DA6401 - Assignment 3

This repository contains the files for the third assignment of the course Deep Learning at IIT Madras

Implemented a Encoder Decoder Architecture with and without Attention Mechanism and used then to perform Transliteration on the Akshanrankar Dataset provided. These models where built using RNN, LSTM and GRU cells provided by PyTorch.

Jump to Section: Usage

Report: Report
Encoder

The encoder is a simple cell of either LSTM, RNN or GRU. The input to the encoder is a sequence of characters and the output is a sequence of hidden states. The hidden state of the last time step is used as the context vector for the decoder.
Decoder

The decoder is again a simple cell of either LSTM, RNN or GRU. The input to the decoder is the hidden state of the encoder and the output of the previous time step. The output of the decoder is a sequence of characters. The decoder has an additional fully connected layer and a log softmax which is used to predict the next character.
Attention Mechanism

The attention mechanism is implemented using the dot product attention mechanism. The attentions are calulated by a weighted sum of softmax values of dot products of the hidden states of the decoder and the hidden states of the encoder. The attention values are then concatenated with the hidden states of the decoder and passed through a fully connected layer to get the output of the decoder.
Dataset

The dataset used is the Aksharankar Dataset provided by the course. The dataset contains 3 files, namely, train.csv, valid.csv and test.csv for each language for a subset of indian languages. I have used the Tamil dataset for this assignment. The dataset contains 2 columns, namely, English and Tamil words which are the input and output strings respectively.
Used Python Libraries and Version

    Python 3.10.9
    Pytorch 1.13.1
    Pandas 1.5.3

Usage

To run the training code for the standard encoder decoder architecture using the best set of hyperparameters, run the following command:

python3 train-without-attention.py

To run the training code for the encoder decoder architecture with attention mechanism using the best set of hyperparameters, run the following command:

python3 train-with-attention.py


To run with custom hyperparameters, run the following command:


# The output of the above command is as follows:
usage: train-without-attention.py [-h] 
                [-es EMBED_SIZE] 
                [-hs HIDDEN_SIZE] 
                [-ct CELL_TYPE] 
                [-nl NUM_LAYERS] 
                [-d DROPOUT]
                [-lr LEARNING_RATE] 
                [-o OPTIMIZER] 
                [-l LANGUAGE]

Transliteration Model

options:
  -h, --help                                        show this help message and exit
  -es EMBED_SIZE, --embed_size EMBED_SIZE           Embedding Size, good_choices = [8, 16, 32]
  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE        Hidden Size, good_choices = [128, 256, 512]
  -ct CELL_TYPE, --cell_type CELL_TYPE              Cell Type, choices: [LSTM, GRU, RNN]
  -nl NUM_LAYERS, --num_layers NUM_LAYERS           Number of Layers, choices: [1, 2, 3]
  -d DROPOUT, --dropout DROPOUT                     Dropout, good_choices: [0, 0.1, 0.2]
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE  Learning Rate, good_choices: [0.0005, 0.001, 0.005]
  -o OPTIMIZER, --optimizer OPTIMIZER               Optimizer, choices: [SGD, ADAM]
  -l LANGUAGE, --language LANGUAGE                  Language

To run the training code for the attention mechanism with custom hyperparameters, run the following command:

python3 train_attention.py -h

usage: train-with-attention.py [-h] 
                          [-es EMBED_SIZE] 
                          [-hs HIDDEN_SIZE] 
                          [-ct CELL_TYPE] 
                          [-nl NUM_LAYERS]
                          [-dr DROPOUT] 
                          [-lr LEARNING_RATE] 
                          [-op OPTIMIZER] 
                          [-wd WEIGHT_DECAY]
                          [-l LANG]

Transliteration Model with Attention

options:
  -h, --help                                        show this help message and exit
  -es EMBED_SIZE, --embed_size EMBED_SIZE           Embedding size
  -hs HIDDEN_SIZE, --hidden_size HIDDEN_SIZE        Hidden size
  -ct CELL_TYPE, --cell_type CELL_TYPE              Cell type
  -nl NUM_LAYERS, --num_layers NUM_LAYERS           Number of layers
  -dr DROPOUT, --dropout DROPOUT                    Dropout
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE  Learning rate
  -op OPTIMIZER, --optimizer OPTIMIZER              Optimizer
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY     Weight decay
  -l LANG, --lang LANG                              Language
