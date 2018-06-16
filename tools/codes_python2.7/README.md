# Text summarization

- A Survey on the Abstractive Summarization with Sequence-to-Sequence models
- Bug report tshi at vt dot edu.

## Requirements

- Python 2.7
- glob
- argparse
- shutil
- Pytorch 0.3.0.post4

Use following scripts to setup
- [AWS Tesla K80](https://github.com/tshi04/seq2seq_coverage_ST/tree/master/tools/CONFIG)
- [pyrouge and ROUGE-1.5.5](https://github.com/tshi04/textsum_ST/tree/master/tools/ROUGE)

## Usuage

Please set bool type parameters in the main.py file.

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

- The following combinations failed during the training after several epochs.
```
concat + temporal
concat + temporal + attn_decoder
```
- We have tried to optimize the memory usage, but we are still not quite happy with it.
- Merge the LSTM and GRU decoders.

## Git References

- https://github.com/abisee/pointer-generator.git
- https://github.com/MaximumEntropy/Seq2Seq-PyTorch
- https://github.com/OpenNMT/OpenNMT
- https://github.com/spro/practical-pytorch
