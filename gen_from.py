###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import numpy as np
import torch
from torch.autograd import Variable

import data
import codecs

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)
hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
if args.cuda:
    input.data = input.data.cuda()

input_data = None
with codecs.open("input.txt", encoding='utf-8') as f:
#	input_data = f.read().replace("\n"," ")
	input_data = f.read()

top = 5
for i in input_data:
    try:
        idx = corpus.dictionary.word2idx[i] 	
    except:
        idx = corpus.dictionary.word2idx[" "] 	
        pass
    try:
        if i == '\n':
            hidden = model.init_hidden(1)
            word_weights = None
            print ""
            continue
        else:
            v,ind =  torch.sort(word_weights)
            norm = v / torch.sum(v)
            xx = -torch.sum(norm * torch.log(norm) / np.log(2))
            rank = ind.index(idx)[0]  
#            print "%s@%.10f|" % (i,word_weights[idx] / word_weights.sum()),
            print "%s@%.10f|" % (i,xx),
#            print "%s@%.10f|" % (i, rank * 1.0 / len(v)),
            for t in range(1,top + 1):
                s = corpus.dictionary.idx2word[ind[-t]]
                if s == '\n':
                    s = " "
                print "%s@%.10f|" % (s , v[-t] / word_weights.sum()),
            print "" 
#            print "%s,%.5f" % (i,word_weights[idx] / word_weights.sum()),
    except:
        print "%s@1.0|" % (i),
        for t in range(1,top + 1):
            print "%s@%.10f|" %(" ",0.0),
        print ""
        pass
    input.data.fill_(idx)
    output, hidden = model(input, hidden)
    word_weights = output.squeeze().data.div(args.temperature).exp().cpu()

