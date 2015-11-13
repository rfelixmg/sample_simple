# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:11:38 2015

@author: felix
"""

import numpy as np
from StringIO import StringIO
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

#ensolarado = 1; chuvoso = 2; nublado = 3;
#quente = 1; frio =2; morna=3;
#alta =1; normal =2;
#verdadeo = 1; falso =0;
#sim = 1; nao =0
y1_test = np.array([1,1,2,1])
Y1_TRAIN = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
X1_TRAIN = np.array([[1, 1, 1, 0], #1
              [1, 1, 1, 1], #2
              [2, 2, 2, 1], #3
              [1, 3, 1, 0], #4
              [2, 3, 1, 1], #5
              [3, 1, 1, 0], #6
              [2, 3, 1, 0], #7
              [2, 2, 2, 0], #8
              [3, 2, 2, 1], #9
              [1, 2, 2, 0], #10
              [2, 3, 2, 0], #11
              [1, 3, 2, 1], #12
              [3, 3, 1, 1], #13
              [3, 1, 2, 0]]) #14
              



y2_test = np.array([1,1,1,2])
y3_test = np.array([1,1,3,4])
Y2_TRAIN = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
X2_TRAIN =np.array([[0, 1, 1, 1], #1
					[1, 0, 1, 2], #2
					[0, 0, 2, 3], #3
					[1, 0, 1, 3], #4
					[1, 1, 1, 4], #5
					[0, 0, 2, 1], #6
					[0, 1, 3, 2], #7
					[0, 0, 3, 3], #8
					[1, 1, 1, 1], #9
					[0, 1, 3, 4], #10
					[0, 1, 3, 1], #11
					[1, 1, 1, 3] ])#12
     
def main():
    gnb = GaussianNB()
    dt = tree.DecisionTreeClassifier(criterion="entropy")
    neigh = KNeighborsClassifier(n_neighbors=3)
    
    #y_pred = neigh.fit(X2_TRAIN, Y2_TRAIN).predict(y3_test)

    y_pred = gnb.fit(X1_TRAIN, Y1_TRAIN).predict(y1_test)
    
    print "Test 1 class: ", y_pred
    print "Prob 1 class: ", gnb.fit(X1_TRAIN, Y1_TRAIN).predict_proba(y1_test)
    
    #clf = dt.fit(X2_TRAIN, Y2_TRAIN)
    #out = StringIO()
    #out = tree.export_graphviz(clf, out_file=out)
    #print out.getvalue()
    #y_pred = clf.predict(y2_test)
    #print "Test 2 class: ", y_pred
    print "Test 3 class: ", y_pred
    
if __name__ == '__main__':
    main()
