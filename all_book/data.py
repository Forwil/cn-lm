#-*- coding:utf-8 -*-
import os
import torch
import sys
import codecs
import json
reload(sys)
sys.setdefaultencoding('utf-8')

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def to_json(self, aft = ""):
        f = open("word2idx" + aft + ".txt","w")
        f.write(json.dumps(self.word2idx))
        f.close()   
        f = open("idx2word" + aft + ".txt","w")
        f.write(json.dumps(self.idx2word))
        f.close()   

    def from_json(self,aft = ""):
        f = open("word2idx" + aft + ".txt","r")
        self.word2idx = json.loads(f.read())
        f.close()   
        f = open("idx2word" + aft + ".txt","r")
        self.idx2word = json.loads(f.read())
        f.close()   


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        if len(path) == 0 or path[0] == "_":
            self.dictionary.from_json(path)
            return
#        return  
       # print "res from"
        print "init train.txt"
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        print "init valid.txt"
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        print "init test.txt"
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.dictionary.to_json("_r")

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with codecs.open(path, encoding='utf-8') as f:
            tokens = 0
            for line in f:
                words = list(line) + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with codecs.open(path, encoding='utf-8') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = list(line) + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids
