import os
import sys
from elasticsearch import Elasticsearch, helpers
import sys, json
from os import listdir
import subprocess
import numpy as np

class readData(object):
    def readJSN(self):
        listOfData = []
        for dirpath, dirs, files in os.walk("Data/"):
            for filename in files:
                fname = os.path.join(dirpath, filename)
                if(fname.endswith("json")):
                    with open(fname, encoding="utf8") as myfile:
                        print(fname)
                        data = json.load(myfile)
                        for obj in data:
                            listOfData.append(obj['fields']['DETD'][0])
        return  listOfData


#rd=readData()
#myList=rd.readJSN()
#print(np.array(myList).shape)