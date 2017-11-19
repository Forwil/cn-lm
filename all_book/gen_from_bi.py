###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data
import codecs

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default="",
                    help='aft of table')
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
parser.add_argument('--gpuid' , type=int, default=0,
					help='gpuid')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.gpuid)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
with open("all_book_big_re.pt", 'rb') as f:
    model_rev = torch.load(f)

model.eval()
model_rev.eval()

if args.cuda:
    model.cuda()
    model_rev.cuda()
else:
    model.cpu()
    model_rev.cpu()

corpus = data.Corpus(args.data)
corpus_rev = data.Corpus("_r")

ntokens = len(corpus.dictionary)
ntokens_rev = len(corpus_rev.dictionary)

hidden = model.init_hidden(1)
hidden_rev = model_rev.init_hidden(1)

input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
input_rev = Variable(torch.rand(1, 1).mul(ntokens_rev).long(), volatile=True)

if args.cuda:
    input.data = input.data.cuda()
    input_rev.data = input_rev.data.cuda()

input_data = None

with codecs.open("input.txt", encoding='utf-8') as f:
#	input_data = f.read().replace("\n"," ")
	input_data = f.read().split("\n")

count = 1
top = 10
for l in input_data:
    result = []
    hidden = model.init_hidden(1)
    word_weights = None
    for i in l:
        try:
            idx = corpus.dictionary.word2idx[i] 	
        except:
            idx = corpus.dictionary.word2idx[" "] 	
            pass
        try:
    #        print "%s,%.5f |  %s,%.5f %s,%.5f" % (i,word_weights[idx] / word_weights.sum(), corpus.dictionary.idx2word[ind[-1]], v[-1] / word_weights.sum(), corpus.dictionary.idx2word[ind[-2]], v[-2] / word_weights.sum())
            v,ind =  torch.sort(word_weights)
            x = [i,word_weights[idx] / word_weights.sum()]
            for t in range(1, top + 1):
                s = corpus.dictionary.idx2word[ind[-t]]
                if s == "\n":
                    s = " "
                x.append(s)
            result.append(x)
        except:
            x = [i,1.0]
            for t in range(1,top + 1):
                x.append(" ")
            result.append(x)
            pass
        input.data.fill_(idx)
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()

    hidden_rev = model_rev.init_hidden(1)
    word_weights = None
    l = l[::-1] 
    result_rev = []
    for i in l:
        try:
            idx = corpus_rev.dictionary.word2idx[i] 	
        except:
            idx = corpus_rev.dictionary.word2idx[" "] 	
            pass
        try:
    #        v,ind =  torch.sort(word_weights)
    #        print "%s,%.5f |  %s,%.5f %s,%.5f" % (i,word_weights[idx] / word_weights.sum(), corpus.dictionary.idx2word[ind[-1]], v[-1] / word_weights.sum(), corpus.dictionary.idx2word[ind[-2]], v[-2] / word_weights.sum())
            v,ind =  torch.sort(word_weights)
            x = [i,word_weights[idx] / word_weights.sum()]
            for t in range(1, top + 1):
                s = corpus_rev.dictionary.idx2word[ind[-t]]
                if s == "\n":
                    s = " "
                x.append(s)
            result_rev.append(x)
        except:
            x = [i,1.0]
            for t in range(1,top + 1):
                x.append(" ")
            result_rev.append(x)
            pass
        input_rev.data.fill_(idx)
        output, hidden_rev = model_rev(input_rev, hidden_rev)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()

    result_rev = result_rev[::-1]
    print "ind:"+str(count)
    for index  in range(0,len(result)):
        t = result[index]
        r = result_rev[index]
        if t[0] != r[0]:
            print t,r
        print "%s@%.10f,%.10f" % (t[0],t[1],r[1]),
        for i in range(0,top):
            print "|%s" % (t[2+i]),
        for i in range(0,top):
            print "|%s" % (r[2+i]),
        print ""
    count += 1

