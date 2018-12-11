# Newsroom Dataset

This directory contains the scripts for processing the newsroom dataset.

## Requirements

- Python 3.5
- shutil
- spacy
- pycorenlp and [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)
- nltk

## Usuage

The official website for newsroom dataset is [summari](https://summari.es/).
The newsroom dataset can be obtained from [newsroom git](https://github.com/clic-lab/newsroom).
Following the instructions, we can scrape and extract the dataset.

#### Get plain texts

We can use different packages to tokenize the texts. If you are using 
- ```Stanford CoreNLP``` python3 extract_corenlp.py

- ```spacy``` python3 extract_spacy.py

- ```nltk``` python3 extract_nltk.py


#### Create data and vocabulary

We will create the dataset which will be used as the input of NATS.
```
python3 process_data_vocab.py
```
