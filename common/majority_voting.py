import numpy as np

def majority_voting(L):
    result = []
    seed = 0
    for item in L:
        seed += 1 #for random select
        wl = []
        for c in item:
            if c not in wl:
                wl.append(c)
        if -1 in wl:
            wl.remove(-1)
        count = []
        for c in wl:
            count.append(item.count(c))
        mwl = []
        mc = max(count)
        for i in range(len(wl)):
            if count[i] == mc:
                mwl.append(wl[i])
        np.random.seed(seed) #if there are multiple labels which have the same count then randomly select one
        result.append(mwl[np.random.randint(0, len(mwl))])
    return result
