# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:01:26 2015

@author: felix
"""

#X1,  X2,    X3 ,     X4,    X5, 
Xf = np.array([[180,   8,   3070,   1300,  3504],
              [150,   8,   3500,   1650,  3693],   	
              [180,   8,   3180,   1500,  3436],   	
              [160,   8,   3040,   1500,  3433],   
              [170,   8,   3020,   1400,  3449],   
              [150,   8,   4290,   1980,  4341],   
              [140,   8,   4540,   2200,  4354],   
              [140,   8,   4400,   2150,  4312],   
              [140,   8,   4550,   2250,  4425],   
              [150,   8,   3900,   1900,  3850],   
              [150,   8,   3830,   1700,  3563],   
              [140,   8,   3400,   1600,  3609],   
              [150,   8,   4000,   1500,  3761],   
              [140,   8,   4550,   2250,  3086],   
              [240,   4,   1130,   9500,  2372],   
              [220,   6,   1980,   9500,  2833],   
              [180,   6,   1990,   9700,  2774],   
              [210,   6,   2000,   8500,  2587],   
              [270,   4,   9700,   8800,  2130], 
              [260,   4,   9700,   4600,  1835]])

yf = np.array([120, 115, 110, 120, 105, 100,  90,  85, 100,  85, 100,  80,  95, 100, 150, 155, 155, 160, 145, 205])
              
from sklearn import linear_model, tree, cross_validation, neighbors

#clf = linear_model.LinearRegression()
#clf = neighbors.KNeighborsRegressor(5)
clf = tree.DecisionTreeRegressor(max_depth=3)

loo = cross_validation.LeaveOneOut(14)
len(loo)

def function6():
    result = {'mean_absolute_error':[], 
          'mean_squared_error':[], 
          'median_absolute_error':[]}

    for train_index, test_index in loo:
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = Xf[train_index], Xf[test_index]
        y_train, y_test = yf[train_index], yf[test_index]
        
        
        clf.fit(X_train, y_train)
        y_result = clf.predict(X_test)
        result.get('mean_absolute_error').append(metrics.mean_absolute_error(y_test, y_result))        
        result.get('mean_squared_error').append(metrics.mean_squared_error(y_test, y_result))        
        result.get('median_absolute_error').append(metrics.mean_squared_error(y_test, y_result))        
        #result.get('r2').append(metrics.r2_score(y_test, y_result)) 
        
    print "-------------------------------"
    print "mean_absolute_error: ", np.mean(result.get('mean_absolute_error'))
    print "mean_squared_error: ", np.mean(result.get('mean_squared_error'))
    print "median_absolute_error: ", np.mean(result.get('median_absolute_error'))
    
def main():
    y_test = np.array([245,4,9700,4600,1835])
    clf.fit(Xf, yf)
    y_pred = clf.predict(y_test)
    print "Result test: ", y_pred

def function9():
    y_test = np.array([245,4,9700,4600,1835])
    clf.fit(Xf, yf)
    y_pred = clf.predict(y_test)
    print "Result test: ", y_pred

    
if __name__ == '__main__':
    main()