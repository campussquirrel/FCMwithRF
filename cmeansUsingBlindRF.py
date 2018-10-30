from sklearn.feature_extraction.text import TfidfVectorizer
import Preprocessing
import numpy as np
from os import path
import sys
#sys.path.append(path.abspath('PossibilisticCMeans-master/'))
from plot import plot
import cmeans
from main import  verify_clusters
from readJSON import  readData
import math
#f = open('RESULT', 'a', encoding='utf-8')
rd=readData()

text=[]
# list of text documents
"""
text = ["The quick brown fox jumped over the lazy dog.",
		"The lazy dog which jumped fast.",
		"The funny jumping dogs lenses eyes contact weak.",
		"quick brown fox is mine!",
        "contact lenses are for weak eyes",
        "eye problems using lenses",
        "brown fox is mine."
        "'\n\t\t>\n\t\t\n\t\t\nBACKGROUND OF THE INVENTION\n"]
"""
text=rd.readJSN()

def textCollection2vectorCollection(fit,transform):
    forfit=fit
    forTransform=transform
    # create the transform
    vectorizer = TfidfVectorizer(tokenizer=Preprocessing.NLTKTokenizer(),
                                    strip_accents = 'unicode', # works
                                    stop_words = 'english', # works
                                    lowercase = True, # works
                                    )
    #vectorizer = TfidfVectorizer(lowercase=True, analyzer="word", stop_words="english")
    # tokenize and build vocab
    vectorizer.fit(forfit)

    # summarize
    #print("vectorizer.vocabulary_",vectorizer.vocabulary_)
    #print(vectorizer.idf_)
    # encode document
    #vector = vectorizer.transform([text[0]])
    # summarize encoded vector
    #print(vector.shape)
    #print(vector.toarray())

    vectorMatrix = vectorizer.transform(forTransform)
    DataMatrix=vectorMatrix.toarray()
    return DataMatrix


def clustering(x,c,fuzzifier,error,maxiter):
    v, v0, u, u0, d, t = cmeans.fcm(x.T, c, fuzzifier, error, maxiter)
    #print("v: ",v)
    #print("u: ",np.array(u))
    #print("d: ",d)
    print("t: ",t)
    plot(x, v, u, c)
    return u

def absolutMembershipArray(u,x):
    print("My sample data")
    #verify_clusters(x, c, v, u, labels)
    cluster_membership = np.argmax(u, axis=0)
    print("membership:")
    for i in range(x.shape[0]):
        print(cluster_membership.item(i))




#list of the cluster labels for each document (making listOflists a numpy matrix)
#uMatrix=np.array([np.array(xi) for xi in u])

def document2Clusters(uMatrix,num_samples):
    finaClusteringList=[]
    for i in range(num_samples):
        listOfClusters=[]
        column=uMatrix[:, i]
        #listOfClusters.append(np.argmax(column))
        maxMemValueOfi=np.max(column)
        clus=0
        for value in column:
            if value>0.99*maxMemValueOfi:
                listOfClusters.append(clus)
            clus=clus+1
        finaClusteringList.append(listOfClusters)
    return finaClusteringList

#finaClusteringList=document2Clusters(uMatrix,num_samples)

def clustersList(finaClusteringList,c,num_samples):
    #list of document clusters
    listOfclusterList=[]
    for i in range(c):
        clusterList=[]
        for j in range(num_samples):
            if i in finaClusteringList[j]:
                clusterList.append(j )
        listOfclusterList.append(clusterList)

    return listOfclusterList

#listOfclusterList=clustersList(finaClusteringList,c,num_samples)
#for element in listOfclusterList:
#    print(element)

def dictDocument2Clusters(uMatrix,num_samples):

    dictList=[]

    for i in range(num_samples):

        dictFinaClusteringList = {}
        column=uMatrix[:, i]
        #listOfClusters.append(np.argmax(column))
        maxMemValueOfi=np.max(column)
        clus=0
        for value in column:
            if value>0.99*maxMemValueOfi:
                dictFinaClusteringList[clus]=value

            clus=clus+1

        dictList.append(dictFinaClusteringList)
    return dictList
#dictdoctoCluster=dictDocument2Clusters(uMatrix,num_samples)
#print(dictdoctoCluster)

#list of document clusters with values as list of dict type
def dictClusterList(c,num_samples,dictdoctoCluster):
    myList=[]
    for i in range(c):
        clusterList=[]
        dictClusterList={}
        for j in range(num_samples):
            item=dictdoctoCluster[j]
            keys = np.fromiter(item.keys(), dtype=int)
            if i in keys:
                #clusterList.append(j )
                dictClusterList[j]=item.get(i)#the value whose key is i
        #listOfclusterList.append(clusterList)
        myList.append(dictClusterList)
    return myList

#myList=dictClusterList(c,num_samples,dictdoctoCluster)

#List of top ranked docs for each cluster
def listingTopRankedDocs(myList):
    topRankedList=[]
    for cluster in myList:
        topRankedincluster=[]
        #print(cluster)
        NumOftopRanked=math.ceil(len(cluster) * 0.1)
        sorted_cluster_byValue=sorted(cluster.items(),key=lambda  kv:kv[1],reverse=True)
        for i in range(NumOftopRanked):
            topRankedincluster.append(sorted_cluster_byValue[i])
        topRankedList.append(topRankedincluster)
    return topRankedList

#topRankedList=listingTopRankedDocs(myList)
#print(topRankedList)

def RelevanceFeedbackDocs(topRankedList):
    RFlist=[]
    for item in topRankedList:
        for doc in item:
            RFlist.append(doc[0])

    RFdocuments=[text[i] for i in RFlist]
    return RFdocuments




DataMatrix=textCollection2vectorCollection(text,text)
""""Here we test out sample data"""
num_samples = DataMatrix.shape[0]
#num_features = 8
c = 23
fuzzifier = 1.2
error = 0.001
maxiter = 200
mylabels=labels = np.array([1,0,0,1,2,2,1])
x= DataMatrix#generate_data(num_samples, num_features, c, shuffle=False)



def runFunction(x,c,fuzzifier,error,maxiter,num_samples):
    u=clustering(x,c,fuzzifier,error,maxiter)
    uMatrix=np.array([np.array(xi) for xi in u])
    finaClusteringList=document2Clusters(uMatrix,num_samples)
    listOfclusterList=clustersList(finaClusteringList,c,num_samples)
    for element in listOfclusterList:
        print(element)

    dictdoctoCluster=dictDocument2Clusters(uMatrix,num_samples)
    #print(dictdoctoCluster)
    myList=dictClusterList(c,num_samples,dictdoctoCluster)
    topRankedList=listingTopRankedDocs(myList)
    #print(topRankedList)
    RFdocuments=RelevanceFeedbackDocs(topRankedList)
    #print(RFdocuments)
    return  RFdocuments

RFdocuments=runFunction(x,c,fuzzifier,error,maxiter,num_samples)
#SECOND ROUND--CLUSTERS AFTER RF
print("SECOND ROUND--CLUSTERS AFTER RF")
DataMatrix=textCollection2vectorCollection(RFdocuments,text)
""""Here we test out sample data"""
num_samples = DataMatrix.shape[0]
#num_features = 8

x = DataMatrix#generate_data(num_samples, num_features, c, shuffle=False)
runFunction(x,c,fuzzifier,error,maxiter,num_samples)

#f.close()