# Text summarization

Abstractive Text Summarization: A Review.

### Package Required
1. Pytorch 0.3.0.post4
Please refer these two scripts: https://github.com/tshi04/seq2seq_coverage_ST/tree/master/tools/CONFIG


### Note



### Some Limitations

1. For the pointer-generator network, we cannot handle the case when the vocabulary for source text and summaries are different.
2. Since we test some coverage mechanisms, the length of source text and summaries are fixed in the beginning. It is not flexiable.
3. When using the argmentation like: 
    python main.py --pointer_net False
   It will still pointer_net=True. 
   It is highly recommended to make changes to the default value of main.py and run python main.py
   
### Work to do

1. In the beam search, add a function that can copy the word to replace unk.
2. In the attention part, add a function that can attend the decoder part as well.

### Git Referenced

- https://github.com/abisee/pointer-generator.git
- https://github.com/MaximumEntropy/Seq2Seq-PyTorch
- https://github.com/OpenNMT/OpenNMT
- https://github.com/spro/practical-pytorch

And more.