import random
import numpy as np

ytrain=np.array([[1,-1,1,-1,-1,+1]]).T



xtrain=np.array(
[[1.2, 0.7,1.],
[-0.3, 0.5,1.],
[-3., -1.,1.] ,
[0.1, 1.0,1.],
[3.0, 1.1,1.],
[2.1, -3,1.]] )

for i in xrange(400):
    pick=np.random.randint(0,len(xtrain)-1)


    l1=np.random.random((3,1))-.5
    l2=np.random.random((3,1))-.5
    l3=np.random.random((3,1))-.5

    out=np.random.random((1,4))-.5



    n1= max(0.,float(xtrain[pick].dot(l1)))
    n2 = max(0., float(xtrain[pick].dot(l2)))
    n3 = max(0., float(xtrain[pick].dot(l3)))
    neurons=np.array([[n1,n2,n3]])
    score=np.sum(np.array([[n1,n2,n3,1]])*out)
    pull=0.

    if ytrain[pick]==1 and score<1.:
        pull=+1
    if ytrain[pick]==-1 and score >1:
        pull=-1

    dscore=pull*1.

    dout=np.array([[n1,n2,n3,1]])*dscore

    dneurons=out[0,:-1]*dscore


    dneurons=dneurons*(neurons!=0)

    dabc=xtrain[pick]*dneurons

    dneurons+=-neurons*np.array([[1.,1.,0.]])

    stepsize=.01

    neurons+=stepsize*dneurons
    out+=stepsize*dout

print neurons
print out


    for i in xrange(len(ytrain)):

    n1= max(0.,float(xtrain[pick].dot(l1)))
    n2 = max(0., float(xtrain[pick].dot(l2)))
    n3 = max(0., float(xtrain[pick].dot(l3)))
    neurons=np.array([[n1,n2,n3]])
    score=np.sum(np.array([[n1,n2,n3,1]])*out)

