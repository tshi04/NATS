# NATS toolkit

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/tshi04/NATS/blob/master/LICENSE)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/tshi04/NATS/graphs/contributors)
[![image](https://img.shields.io/github/issues/Naereen/StrapDown.js.svg)](https://github.com/tshi04/NATS/issues)
[![image](https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat)](https://arxiv.org/abs/1812.02303)

#### Our new implementation is available at https://github.com/tshi04/LeafNATS. Please check it.

- Check python2.7 version of NATS from [here](https://github.com/tshi04/textsum/tree/master/tools/NATS2.7).
- This repository is a pytorch implementation of seq2seq models for the following survey:

#### [Abstractive Text Summarization with Sequence-to-Sequence Models](https://arxiv.org/abs/1812.02303)
[Tian Shi](https://github.com/tshi04), 
[Yaser Keneshloo](https://github.com/yaserkl), 
[Naren Ramakrishnan](http://people.cs.vt.edu/naren/), 
[Chandan K. Reddy](http://people.cs.vt.edu/~reddy/)


## Requirements and Installation

- Python 3.5.2
- glob
- argparse
- shutil
- pytorch 1.0

**Use following scripts to**

- [Set up GPU, cuda and pytorch](https://github.com/tshi04/NATS/tree/master/tools/config_server)
- [Install pyrouge and ROUGE-1.5.5](https://github.com/tshi04/textsum/tree/master/tools/rouge_package)

## Dataset

In this survey, we run an extensive set of experiments with NATS on the following datasets. Here, we provide the link to CNN/Daily Mail dataset and data processing codes for Newsroom and Bytecup2018 datasets. 
- [CNN/Daily Mail](https://github.com/abisee/pointer-generator)
- [Newsroom](https://github.com/tshi04/textsum/tree/master/tools/newsroom_process)
- [Bytecup2018](https://github.com/tshi04/NATS/tree/master/tools/bytecup_process)

In the dataset, \<s\> and \</s\> is used to separate sentences. \<sec\> is used to separate summaries and articles. We did not use the json format because it takes more space and be difficult to transfer.

## Usuage

- ```Training:``` python main.py 

- ```Validate:``` python main.py --task validate

- ```Test:``` python main.py --task beam

- ```Rouge:``` python main.py --task rouge


## Features

The NATS is equipped with following features:

- ```Attention based seq2seq framework.``` 
Encoder and decoder can be LSTM or GRU. The attention scores can be calculated with three different alignment methods.

- ```Pointer-generator network.```

- ```Intra-temporal attention mechanism and intra-decoder attention mechanism.```

- ```Coverage mechanism.```

- ```Weight sharing mechanism.```
Weight sharing mechanism can boost the performance with significantly less parameters.

- ```Beam search algorithm.```
We implemented an efficient beam search algorithm that can also handle cases when batch_size>1.

- ```Unknown words replacement.```
This meta-algorithm can be used along with any attention based seq2seq model.
The OOV words <unk> in summaries are manually replaced with words in source articles using attention weights.

Experimental results can be found in our survey paper.

## Problems and Todos

- Some models failed during the training after several epochs. For example, on CNN/Daily Mail dataset,
```
concat + temporal
concat + temporal + attn_decoder
```
- We have tried to optimize memory usage, but we are still not quite happy with it.
- Merge the LSTM and GRU decoders.







