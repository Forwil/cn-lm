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

import json

f = json.loads(codecs.open(sys.argv[1], encoding='utf-8').read())

thr = 0.00001
if len(sys.argv) >= 3:
    thr = float(sys.argv[2])
thr_pin = 0.0001
if len(sys.argv) >= 4:
    thr_pin = float(sys.argv[3])

def comp(a,b):
    if a[1] < b[1]:
        return 1
    elif a[1] > b[1]:
        return -1
    else:
        return 0
ans = {}
line = 0
for it in f:
    line += 1
    forward = it["for"]
    backward = it["rev"]
    errlist = []
    for i in range(len(forward)): 
        forw = forward[i]
        back = backward[i]
        prob_forw = forw["now"][1]
        prob_back = back["now"][1]
        prob = math.sqrt(prob_forw * prob_back)
        if forw["now"][0] != back["now"][0]:
            print "error", forw["now"][0], back["now"][0]
            exit()
        now = forw["now"][0]
        now_pin = lazy_pinyin(now, errors='ignore') 
        if len(now_pin) > 0:
            now_pin = now_pin[0]
        top = forw["top"] + back["top"]
        top.sort(comp)
        err = []
        if now in [x[0] for x in top]:
            continue
        prob = prob / (top[0][1] + 0.001)
        if prob < thr:
            err = [i, top[0][0]]
        if prob < thr_pin and len(now_pin) > 0:
#            for j in range(len(top)):
#                p = lazy_pinyin(top[j][0], errors = 'ignore')
#                if len(p) > 0 :
#                    p = p[0]
#                dis = distance(now_pin, p )
#                top[j][1] = top[j][1] ** dis
#            top.sort(comp)
#            err = [i,top[0][0]]
            for j in top:
                p = lazy_pinyin(j[0], errors = 'ignore')
                if len(p) > 0 :
                    p = p[0]
                dis = distance(now_pin, p )
                if dis <= 1:
                    err = [i, j[0]]
                    break
        if len(err) > 0:
            errlist.append(err)
    ans[line] = errlist
             
for i in ans:
    print str(i+400).zfill(5),
    if len(ans[i]) == 0:
        print ",0"
    else:
        for j in ans[i]:
            print ",",j[0] + 1,",",j[1],
        print ""
