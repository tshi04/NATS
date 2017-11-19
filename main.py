import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='seq2seq', help='seq2seq | seqGAN')
stc = parser.parse_args()