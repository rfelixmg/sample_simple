# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Users\71412700.MACKLABS.001\.spyder2\.temp.py
"""
import numpy as npr

TID = np.array([[1,0,1,0], [1,1,1,0],[1,0,0,0],[1,0,1,1],[0,0,1,0],[1,1,0,0]])

def apriori(_tid, _s_min):

    (t_shape, i_shape) = _tid.shape 

    support_ = np.array(np.zeros((i_shape, i_shape)), np.float)
    for t_id in range(i_shape):
        X = _tid.transpose()[t_id]
        for a_id in range(i_shape):
            #print "associating att: %i com %i " % (t_id+ 1, a_id+ 1)
            Y = _tid.transpose()[a_id]
            n_XY = 0;            
            for (i_id, x) in enumerate(X):
                y = Y[i_id]                
                if(x and y):
                    n_XY += 1
                support_[t_id, a_id] = n_XY/np.float(t_shape)

    confidence_ = apriori_confidence(support_)

    association_ = confidence_ >= _s_min    
    
    return ( association_, support_, confidence_)
    
def apriori_confidence(_support):

    (s_row, s_col) = _support.shape

    confidence_ = np.array(np.zeros((s_col, s_col)), np.float)   
    
    for s_rid in range(s_row):
        for s_cid in range(s_col):
            #print "confidence at: %i com %i " % (s_rid+ 1, s_cid+ 1)
            s_XY = _support[s_rid, s_cid]
            s_X  = _support[s_rid, s_rid]
            
            #print "%2f / %2f = %2f"%(s_XY,s_X, s_XY/np.float(s_X))

            confidence_[s_rid, s_cid] = s_XY/np.float(s_X)
             
    return confidence_
        
        
        

if __name__ == "__main__":
    prediction = apriori(TID, 0.5)
    
    print "Support: ", prediction[1]
    print "Confidence: ", prediction[2]

    print "Association: ", prediction[0]    
    