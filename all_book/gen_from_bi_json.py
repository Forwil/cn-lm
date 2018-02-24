import argparse

import torch
from torch.autograd import Variable

import data
import codecs
import json

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--checkpoint_rev', type=str, default='./model.pt',
                    help='model checkpoint reverse to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--gpuid' , type=int, default=0,
					help='gpuid')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--input', type=str, default="./input.txt",
                    help='input text')


args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.set_device(args.gpuid)


with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
with open(args.checkpoint_rev, 'rb') as f:
    model_rev = torch.load(f)

model.eval()
model_rev.eval()

if args.cuda:
    model.cuda()
    model_rev.cuda()
else:
    model.cpu()
    model_rev.cpu()

corpus = data.Corpus("_for")
corpus_rev = data.Corpus("_rev")

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

with codecs.open(args.input, encoding='utf-8') as f:
	input_data = f.read().split("\n")

count = 1
top = 10
res = []
res_rev = []
for l in input_data:
    result = []
    hidden = model.init_hidden(1)
    word_weights = None
    for i in l:
        x = {}
        try:
            idx = corpus.dictionary.word2idx[i] 	
        except:
            idx = corpus.dictionary.word2idx[" "] 	
            pass
        try:
            v,ind =  torch.sort(word_weights)
            x["now"] = [i,word_weights[idx] / word_weights.sum()]
            x["top"] = []
            for t in range(1, top + 1):
                s = corpus.dictionary.idx2word[ind[-t]]
                if s == "\n":
                    s = " "
                x["top"].append([s,v[-t] / v.sum()])
            result.append(x)
        except:
            x["now"] = [i,1.0]
            x["top"] = []
            for t in range(1,top + 1):
                x["top"].append([" ",0])
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
        x = {}
        try:
            idx = corpus_rev.dictionary.word2idx[i] 	
        except:
            idx = corpus_rev.dictionary.word2idx[" "] 	
            pass
        try:
            v,ind =  torch.sort(word_weights)
            x["now"] = [i,word_weights[idx] / word_weights.sum()]
            x["top"] = []
            for t in range(1, top + 1):
                s = corpus_rev.dictionary.idx2word[ind[-t]]
                if s == "\n":
                    s = " "
                x["top"].append((s,v[-t] / v.sum()))
            result_rev.append(x)
        except:
            x["now"] = [i,1.0]
            x["top"] = []
            for t in range(1,top + 1):
                x["top"].append([" ",0])
            result_rev.append(x)
            pass
        input_rev.data.fill_(idx)
        output, hidden_rev = model_rev(input_rev, hidden_rev)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()

    result_rev = result_rev[::-1]
    res.append(result)
    res_rev.append(result_rev)
    count += 1
    print count

res_all = []
for i in range(0,len(res)-1):
    t = {}
    t["for"] = res[i]
    t["rev"] = res_rev[i]
    res_all.append(t)
f = open(args.outf,"w")
f.write(json.dumps(res_all, indent = 4))
f.close()
