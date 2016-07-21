#!/usr/bin/env python

import sys
import random

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-n", "--model-num", action="store", dest="n_model",
                  help="number of models to train", type="int")
parser.add_option("-r", "--sample-ratio", action="store", dest="ratio",
                  help="ratio to sample for each ensemble", type="float")

options, args = parser.parse_args(sys.argv)

r = options.ratio
M = options.n_model


for line in sys.stdin:
        value = line.strip()
        for i in range(0,M):
            m = random.random()
            if m < r:
                if len(value) > 0:
                    print "%d\t%s" % (i, value)





#for i in range(0,M):
#    m = random.random()
#    if m < r:
#        for line in sys.stdin:
#            key = random.randint(0, options.n_model - 1)
#            value = line.strip()
#            if len(value) > 0:
#                print "%d\t%s" % (i, value)




#for line in sys.stdin:
#    key = random.randint(0, options.n_model - 1)
#    value = line.strip()
#    if len(value) > 0:
#        print "%d\t%s" % (key, value)
