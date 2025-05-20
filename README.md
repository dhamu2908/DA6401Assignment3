🧠 DA6401 - Assignment 3: Transliteration using Encoder-Decoder Models

[Report](https://wandb.ai/m_dhamu2908/Deep_Learning_Assignment3/reports/DA6401-Assignment-3-Report--VmlldzoxMjg2MDM1NQ?accessToken=ghrm4p1s3lhvgofh88hzbvb5oywcih8tygoi3yuo2c3wschma0tu28q4iqd5mmje)


This repository contains the implementation for Assignment 3 of the Deep Learning course at IIT Madras.

We explore sequence-to-sequence models using Encoder-Decoder architectures, both with and without Attention mechanisms, for the task of transliteration on the Aksharantar dataset.

📌 Table of Contents
Overview

Model Architectures

Encoder

Decoder

Attention Mechanism

Dataset

Dependencies

Usage

Standard Encoder-Decoder

Encoder-Decoder with Attention

Custom Training Options

📖 Overview
The objective is to build a neural machine transliteration system using:

RNN, LSTM, and GRU cells (via PyTorch)

Optional attention mechanism for improved performance

🏗️ Model Architectures
🔹 Encoder
The encoder is a recurrent layer (LSTM, GRU, or RNN) that takes a sequence of input characters (e.g., English word) and returns a sequence of hidden states.
The final hidden state is used as the context vector for the decoder.

🔹 Decoder
The decoder also uses RNN-based cells and generates the output sequence (e.g., Tamil word) character by character.
Key components:

RNN/LSTM/GRU cells

Fully connected layer

Log Softmax for prediction

🔹 Attention Mechanism
Implemented using dot-product attention.
The attention scores are computed as the dot-product between encoder and decoder hidden states, followed by a softmax. The attention vector is:

Combined with the decoder’s hidden state

Passed through a linear layer to produce the final output at each step

📂 Dataset
Aksharantar Dataset provided by the course

Files:

train.csv

valid.csv

test.csv

Each file contains:

English (input string)

Telugu (target string)

Language Used: Telugu

🧰 Dependencies
Ensure you are using:

bash
Copy
Edit
Python      3.10.9  
PyTorch     1.13.1  
Pandas      1.5.3
Install dependencies using:

bash
Copy
Edit
pip install torch==1.13.1 pandas==1.5.3
🚀 Usage
▶️ Standard Encoder-Decoder
To train the model without attention, run:

bash
Copy
Edit
python3 train-without-attention.py
▶️ Encoder-Decoder with Attention
To train the model with attention, run:

bash
Copy
Edit
python3 train-with-attention.py
⚙️ Custom Training Options
You can provide hyperparameters via command-line flags:

🔧 Without Attention
bash
Copy
Edit
python3 train-without-attention.py -h
Available Options:

Flag	Description	Choices
-es, --embed_size	Embedding size	8, 16, 32
-hs, --hidden_size	Hidden state size	128, 256, 512
-ct, --cell_type	RNN cell type	LSTM, GRU, RNN
-nl, --num_layers	Number of layers	1, 2, 3
-d, --dropout	Dropout	0, 0.1, 0.2
-lr, --learning_rate	Learning rate	0.0005, 0.001, 0.005
-o, --optimizer	Optimizer	SGD, ADAM
-l, --language	Language	e.g., Telugu

🔧 With Attention
bash
Copy
Edit
python3 train-with-attention.py -h
Available Options:

Flag	Description
-es, --embed_size	Embedding size
-hs, --hidden_size	Hidden state size
-ct, --cell_type	RNN cell type
-nl, --num_layers	Number of layers
-dr, --dropout	Dropout
-lr, --learning_rate	Learning rate
-op, --optimizer	Optimizer
-wd, --weight_decay	Weight decay
-l, --lang	Language

📌 Notes
Ensure that the train.csv, valid.csv, and test.csv files are placed in the correct directory.

Evaluate models using validation/test set performance after training.

📚 Report
Refer to Report for a detailed write-up of:

Model architecture

Training methodology

Hyperparameter tuning

Results and conclusions
