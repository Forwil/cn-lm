import sys
import os
minthr = 0.000001
minthr_det = 0.000001

cmd_mod = "python deal.py top5.txt  %f %f > o"
while True:
    minpinthr = minthr
    minthr_det = 0.000001    
    while True:
        cmd = cmd_mod % (minthr,minpinthr)
        print cmd
        os.system(cmd)
        cmd = "./cp.sh %s" % (str(minthr)+"_"+str(minpinthr))  
        print cmd
        os.system(cmd)
        minpinthr += minthr_det
        if minpinthr > 100 * minthr:
            break
    minthr += minthr_det
    if minthr > 0.001:
        break
     
