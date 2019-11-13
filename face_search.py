import numpy as np
from utils import cosineDistance

def bruteforce(qfe, fdb, threshold):
    """
    glorious for loop through the face database
    """
    min_dst = 100
    best_match = None
    # print("number of imgs: ", len(fdb[0]['feats']))
    # print(fdb[0]['name'])
    for doc in fdb:
        for feat in doc['feats']:
            feat_np = np.asarray(feat)
            # print(feat_np)
            cos_dst = cosineDistance(qfe, feat_np)
            # print(cos_dst)
            if (cos_dst <= threshold) and (cos_dst < min_dst):
                min_dst = cos_dst
                best_match = doc
    return best_match
    
def bruteforce_dep(qfe, fdb, threshold):
    """
    glorious for loop through the face database
    # DEPRECATED
    """
    min_dst = 100
    best_match = None
    for rec in fdb:
        for feat in rec['feats']:
            cos_dst = cosineDistance(qfe, feat)
            print(cos_dst)
            if (cos_dst <= threshold) and (cos_dst < min_dst):
                min_dst = cos_dst
                best_match = rec
    return best_match
    