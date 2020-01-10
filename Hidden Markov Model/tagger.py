import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
    model = None
    states_symbols={}
    for idx, item in enumerate(tags):
        states_symbols[item] = idx
    
    S=len(tags)
    pi=np.zeros((S))
    tag_count={}
    tag_dict={}
    tag_dict_wordcount={}
    obs_dict={}
    
    for i in tags:
        tag_count[i]=0
        tag_dict[i]=[]
        tag_dict_wordcount[i]=[]
    A=np.zeros((S,S))
    l=0
    B=[]
    for j in range (len(train_data)):
        xtag=train_data[j].tags
        xword=train_data[j].words
        ind=states_symbols[train_data[j].tags[0]]
        pi[ind]+=1
        tagencod=np.zeros(len(xtag),dtype='int')
        for i in range(len(xtag)):
            tagencod[i]=tags.index(xtag[i])
        for (m,n) in zip(tagencod,tagencod[1:]):
            A[m][n] += 1 
        for k in range(len(xtag)):
            if xword[k] in obs_dict.keys():
                ind=obs_dict[xword[k]]
                B[ind][states_symbols[xtag[k]]]+=1
            else:
                obs_dict[xword[k]]=l
                temp=np.zeros(S)
                temp[states_symbols[xtag[k]]]=1
                B.append(temp)
                l+=1
#        k=0
#        for i in xtag:
#            if i in tags:
#                if xword[k] in tag_dict[i]:
#                    tag_count[i]+=1
#                    tag_dict_wordcount[i][tag_dict[i].index(xword[k])]+=1
#    #                obseq.append(xword[k])
#                else:
#                    tag_dict[i].append(xword[k])
#                    tag_count[i]+=1
#                    tag_dict_wordcount[i].append(1)
#                    obs_dict[xword[k]]=l
#                    l=l+1
#            k+=1
     
        
#    L=l
#    B=np.zeros((S,L))
#    for m in range(S):
#        k=0
#        for words in tag_dict[list(tag_dict.keys())[m]]:
#            ind= obs_dict[words] 
#            B[m,ind]=tag_dict_wordcount[list(tag_dict.keys())[m]][k]
#            k+=1
    B=np.array(B).T
    bsum=np.sum(B,axis=1).reshape((-1,1))
    B=np.divide(B,bsum, where=bsum!=0)
    asum=np.sum(A,axis=1).reshape((-1,1))
    A=np.divide(A,asum, where=asum!=0)
    k=0
    pi=pi/sum(pi)
#    for i in tags:
#        pi[k]=tag_count[i]/sum(tag_count.values())
#        k+=1
	###################################################
    model = HMM(pi, A, B, obs_dict, states_symbols)
	
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    tagging = []
    S,N=model.B.shape
    for i in range(len(test_data)):
        Osequence=test_data[i].words
        for word in Osequence:
            if word not in model.obs_dict:
                temp=0.000001*np.ones((S,1))
#                print(temp)
                model.B=np.hstack((model.B,temp))
#                B=model.B
                model.obs_dict[word]=len(model.obs_dict)
        tagging.append(model.viterbi(Osequence))
    
	###################################################
	# Edit here
	###################################################
	
    return tagging
