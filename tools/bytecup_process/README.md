# bytecup2018 headline generation


This directory has scripts for processing the bytecup2018 dataset.

## Requirements 

- Python 3.5
- pycorenlp and [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)


## Usuage

The dataset can be obtained from [Byte Cup 2018](https://biendata.com/competition/bytecup2018/).

##### Unzip
```
./unzip_files.sh
```
##### Tokenize
```
python3 tokenize.py --input bytecup.corpus.train.1.txt --output new1.txt
```
#### Process data
```
python3 process_data_vocab.py
```
#### Output
```
train.txt
val.txt
test.txt
vocab
```