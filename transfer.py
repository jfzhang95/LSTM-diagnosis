#!usr/bin/env python
#-*- coding:utf-8 -*-
"""
@author: James Zhang
@date:   2017-03-27
"""


import copy
import numpy as np
import csv

def handle_binary_vector(given_list, k):
    tmp_list = copy.deepcopy(given_list)
    given_list.sort(reverse=True)
    new_sort_array = given_list[0:k]
    index_list = []
    for each_num in new_sort_array:
        index_list.append(tmp_list.index(each_num))
    new_vector_list=np.zeros(len(given_list),dtype='int64')
    for each_position in index_list:
        new_vector_list[each_position]=1
    return (new_vector_list,tmp_list)


PRED = []
with open('testfile.csv', 'rb') as myfile:
    reader = csv.reader(myfile)
    for line in reader:
        PRED.append(line)


print np.array(PRED).shape
