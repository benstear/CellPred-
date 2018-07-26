#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 10:30:02 2018
@author: benstear


DATA SOURCE: Broad Institute Single Cell Portal
STUDY:       Atlas of human blood dendritic cells and monocytes
URL:         https://portals.broadinstitute.org/single_cell/study/atlas-of-human-blood-dendritic-cells-and-monocytes
Total Cells: 1078
"""

import pandas as pd


try: metadata, data
except:
    metadata=pd.read_csv('/Users/dawnstear/downloads/metadata.txt', sep='\t',header=1)
    data=pd.read_csv('/Users/dawnstear/downloads/expression_matrix_tpm.txt',sep='\t',header=0).T
                     
# Get Some metrics on data
# size, use os. and time data upload speed.    
    
# *---------------Preprocessing---------------*
index = list(data.index)    
data.index = range(len(data)) 

if 'TYPE' not in data.columns:   # if not there, insert 'TYPE' col
    data.insert(loc=0, column='TYPE', value=index)


data['Labels'] = data.index # just use index as placeholder
labels = metadata['group']
labels = list(labels)
meta_TYPE = metadata['TYPE']
data_TYPE = data['TYPE']

# *----------- Map cell sample names ('Labels') to proper cell subtype ID  ('Group') ------*
for i in range(len(metadata)):
    if meta_TYPE[i] == data_TYPE[i]:
        data['Labels'][i] = metadata['group'][i]
    else:
        idx = meta_TYPE[meta_TYPE==data_TYPE[i]].index[0] # python doesnt like this
        data['Labels'][i] = metadata['group'][idx]        # ...A value is trying to be set on a copy of a slice from a DataFrame

# *--------Must convert cell subtype labels to integers for TensorFlow-----*
# ...and also find distrobution of diff cell types
DC1 = DC2 = DC3 = DC4 = DC5 = DC6 = MONO1 = MONO2 = MONO3 = MONO4 = 0
for i in range(len(labels)):
    if labels[i]=='DC1':
        DC1+=1
        labels[i] = 0
    elif labels[i]=='DC2':
        DC2+=1
        labels[i] = 1
    elif labels[i]=='DC3':
        DC3+=1
        labels[i] = 2
    elif labels[i]=='DC4':
        DC4+=1
        labels[i] = 3
    elif labels[i]=='DC5':
        DC5+=1
        labels[i] = 4
    elif labels[i]=='DC6':
        DC6+=1
        labels[i] = 5
    elif labels[i]=='Mono1':
        MONO1+=1
        labels[i] = 6
    elif labels[i]=='Mono2':
        MONO2+=1
        labels[i] = 7
    elif labels[i]=='Mono3':
        MONO3+=1
        labels[i] = 8
    elif labels[i]=='Mono4' :
        MONO4+=1    #  mono3 = 20, but should be 31, maybe datatype copying error ?
        labels[i] = 9
        
celldistro = {'DC1':DC1, 'DC2:':DC2, 'DC3':DC3, 'DC4':DC4, 'DC5':DC5, 'DC6':DC6,
              'Mono1': MONO1, 'Mono2':MONO2, 'Mono3':MONO4, 'Mono4': MONO4}
# Create colored, pie chart of celldistro before and after (truth vs pred) and Venn diagram of genes
        
# Replace new integer labels into old categorical labels
data['Labels'] = labels


#------------reduce number of features just to make sure things are working without computational burden-------------
cols = list(range(10,17590))
#data.drop(data.columns[cols],axis=1,inplace=True)   #combine this & line above

# save data in chop_cellpred desktop folder and load it in model program





