# -*- coding:utf-8 -*
import sys
import torch
reload(sys)
sys.setdefaultencoding('utf-8')
import math
from pypinyin import pinyin,lazy_pinyin
import pypinyin
import codecs

def distance(first,second):  
    if len(first) > len(second):  
        first,second = second,first  
    if len(first) == 0:  
        return len(second)  
    if len(second) == 0:  
        return len(first)  
    first_length = len(first) + 1  
    second_length = len(second) + 1  
    distance_matrix = [range(second_length) for x in range(first_length)]   
    #print distance_matrix  
    for i in range(1,first_length):  
        for j in range(1,second_length):  
            deletion = distance_matrix[i-1][j] + 1  
            insertion = distance_matrix[i][j-1] + 1  
            substitution = distance_matrix[i-1][j-1]  
            if first[i-1] != second[j-1]:  
                substitution += 1  
            distance_matrix[i][j] = min(insertion,deletion,substitution)  
    return distance_matrix[first_length-1][second_length-1]  

f = codecs.open(sys.argv[1], encoding='utf-8').read().split("\n")[:-1]
thr = 0.00001
if len(sys.argv) >= 3:
    thr = float(sys.argv[2])
thr_pin = 0.0001
if len(sys.argv) >= 4:
    thr_pin = float(sys.argv[3])

lines = 1
count = 0
err = 0

ans = [ 0 ] * 1101
ans[1] = []

#model = torch.load("result.pt")
cl = None

for i in f:
    if "ind:" in i:
#        cl = [1,1,1,1,1,1]
        lines = int(i.split(":")[1])
        ans[lines] = []
        count = 0
        continue
    c = [j.strip() for j in i.split("|")[:-1]]
    if(len(c) == 0):
        continue
    count += 1

    now = c[0][:-1].split("@")[0]
    prob = float(c[0][:-1].split("@")[1].split(",")[0])
    prob_rev = float(c[0][:-1].split("@")[1].split(",")[1])
#    cl = cl[2:] + [prob,prob_rev]
    
    now_pin = lazy_pinyin(now,errors='ignore')
    top5 = []
    for j in c[2:]:
        top5.append((j))
    top5_pin = []
    for j in top5:
        top5_pin.append(lazy_pinyin(j,errors='ignore'))
#    s = 2 * (prob * prob_rev) / (prob + prob_rev)
    
#    prob = cl[2]
#    prob_rev = cl[3]
    s = math.sqrt(prob * prob_rev) 
#    p = model.forward(torch.autograd.Variable(torch.Tensor([cl])))[0].data[0]
#    print prob,prob_rev,p
#    s = 1 - p
    if s < thr:
#    if prob  < thr:
       # if p < 0.5:
       #     continue
        ans[lines].append(count)
        err += 1
        continue
    if  s < thr_pin and len(now_pin)>0:
        for j in top5_pin:
            if len(j)== 0:
                continue;
            dis = distance(j[0],now_pin[0])
            if dis <= 0:
                ans[lines].append(count)
                break

for i in range(1,1001):
    print str(i).zfill(4),
    if(len(ans[i]) == 0):
        print ",0"
    else:
        for j in ans[i]:
            print ",",j,
        print ""
