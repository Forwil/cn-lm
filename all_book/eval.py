import torch
import model
import data
import sys
from torch.autograd import Variable

import torch.nn as nn
import math
torch.manual_seed(1111)
torch.cuda.manual_seed(1111)
torch.cuda.set_device(0)
corpus = data.Corpus("../data/wiki")

eval_batch_size = 10

criterion = nn.CrossEntropyLoss()

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    data = data.cuda()
    return data

def get_batch(source, i, evaluation=False):
    seq_len = min(35, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, 35):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
        l = total_loss[0] /(i + 1) 
        print l,math.exp(l),i, data_source.size(0)-  1 
    return total_loss[0] / len(data_source)


val_data = batchify(corpus.valid, eval_batch_size)

ntokens = len(corpus.dictionary)

model = torch.load(sys.argv[1])
model.eval()

model.cuda()
torch.cuda.set_device(0)

val_loss = evaluate(val_data)
print val_loss
