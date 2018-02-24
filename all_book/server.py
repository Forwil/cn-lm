import web
import json
import codecs
import hashlib
import numpy as np
import torch
from torch.autograd import Variable
import data
import codecs

urls = (
    '/', 'index',
    '/check', 'check'
)

render = web.template.render('templates/',)

def comp(a,b):
    if a[1] < b[1]:
        return 1
    elif a[1] > b[1]:
        return -1
    else:
        return 0

class check:
    def GET(self):
        return "123"

    def POST(self):
        i = web.input()
        ret = getjson(i["content"])
        all  = []
        l = []
        for i in ret:
            forw = i["for"]
            back = i["rev"]
            for c in range(len(forw)):
                obj = {}
                obj["name"] = forw[c]["now"][0]
                obj["prob"] = [forw[c]["now"][1] , back[c]["now"][1]]
                top = forw[c]["top"] + back[c]["top"]
                top.sort(comp)
                obj["top"] = top
                l.append(obj)
                if len(l) > 20:
                    all.append(l)
                    l = []        
        if len(l) > 0:
            all.append(l)
                     
        return json.dumps(all)



class index:
    def GET(self):
        web.header("Content-Type","text/html; charset=utf-8")
        return render.check()

    def POST(self):
        return json.dump([123])
    

checkpoint = "big.pt"
checkpoint_rev = "big_re.pt"
cuda = True
gpuid = 0
temperature = 1
torch.cuda.set_device(gpuid)
with open(checkpoint, 'rb') as f:
    model = torch.load(f)
with open(checkpoint_rev, 'rb') as f:
    model_rev = torch.load(f)

model.eval()
model_rev.eval()

model.cuda()
model_rev.cuda()

corpus = data.Corpus("_for")
corpus_rev = data.Corpus("_rev")

ntokens = len(corpus.dictionary)
ntokens_rev = len(corpus_rev.dictionary)

hidden = model.init_hidden(1)
hidden_rev = model_rev.init_hidden(1)

input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
input_rev = Variable(torch.rand(1, 1).mul(ntokens_rev).long(), volatile=True)

if cuda:
   input.data = input.data.cuda()
   input_rev.data = input_rev.data.cuda()

def getjson(input_data):
   global model
   global model_rev
   global corpus
   global corpus_rev
   global ntokens
   global ntokens_rev
   global hidden
   global hidden_rev
   global input
   global input_rev
 
   count = 1
   top = 10
   res = []
   res_rev = []
   input_data = [input_data]
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
           word_weights = output.squeeze().data.div(temperature).exp().cpu()

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
           word_weights = output.squeeze().data.div(temperature).exp().cpu()

       result_rev = result_rev[::-1]
       res.append(result)
       res_rev.append(result_rev)
       count += 1
   res_all = []
   for i in range(0,len(res)):
       t = {}
       t["for"] = res[i]
       t["rev"] = res_rev[i]
       res_all.append(t)
   return res_all


if __name__ == "__main__":
    
   app = web.application(urls, globals()) 
   app.run()
