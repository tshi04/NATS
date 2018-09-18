# Text summarization

- This is a Pytorch implementation of the seq2seq models for abstractive summarization.
- Previous version with python2.7 can be found [here](https://github.com/tshi04/textsum/tree/master/tools/codes_python2.7). It has not been maintained for a while.
- Neural Abstractive Summarization with Sequence-to-Sequence Models

## Requirements

- Python 3.5.2
- glob
- argparse
- shutil
- pytorch 0.4.0

Use following scripts to setup
- [Tesla K80](https://github.com/tshi04/textsum/tree/master/tools/config_server)
- [Tesla V100](https://github.com/tshi04/SetEC2)
- [pyrouge and ROUGE-1.5.5](https://github.com/tshi04/textsum/tree/master/tools/rouge_package)

## Usuage

#### Dataset

- [CNN/Daily Mail](https://github.com/abisee/pointer-generator)
- [newsroom](https://github.com/tshi04/textsum/tree/master/tools/newsroom_process)

#### Training
```
python main.py 
```
#### Validate
```
python main.py --task validate --batch_size 10
```
#### Test
```
python main.py --task beam --batch_size 4
```
#### Rouge
```
python main.py --task rouge
run cal_rouge.ipynb to calculate the rouge score using pyrouge and ROUGE-1.5.5
```

## Problems and Todos

- Some models failed during the training after several epochs. For example, on CNN/Daily Mail dataset,
```
concat + temporal
concat + temporal + attn_decoder
```
- We have tried to optimize the memory usage, but we are still not quite happy with it.
- Merge the LSTM and GRU decoders.
- textsum has been written from the scratches. If you find any bug, please contact Tian by tshi at vt dot edu.
