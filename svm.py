import numpy as np
import random
class Unit(object):

    def __init__(self,value,gradient=0):
        self.value=value
        self.gradient=gradient
    def __str__(self):
        return  'value {}  gradient {} '.format(self.value,self.gradient)

class Gate(object):

    def forward(self):
        pass
    def backward(self):
        pass

    def __str__(self):
        return 'u0.value={}  u0.grad={} u1.value={} u1.grad={} output={}'.format(self.u0.value,self.u0.gradient,self.u1.value,self.u1.gradient,self.utop)

class multiplyGate(Gate):

    def forward(self,u0,u1):
        self.u0=u0
        self.u1=u1

        self.utop=Unit(self.u0.value*self.u1.value)

        return self.utop

    def backward(self):
        self.u0.gradient=self.u1.value*self.utop.gradient
        self.u1.gradient=self.u0.value*self.utop.gradient


class addGate(Gate):
    def forward(self,u0,u1):
        self.u0 = u0
        self.u1 = u1
        self.utop=Unit(self.u0.value+self.u1.value)
        return self.utop

    def backward(self):
        self.u0.gradient=1.*self.utop.gradient
        self.u1.gradient=1.*self.utop.gradient

class Circuit(object):
    def __init__(self):
        self.mulg0 = multiplyGate()
        self.mulg1 =multiplyGate()
        self.addg0=addGate()
        self.addg1=addGate()

    def forward(self,x,y,a,b,c):
        '''ax+by+c'''
        self.ax=self.mulg0.forward(a,x)

        self.by=self.mulg1.forward(b,y)



        self.axpby=self.addg0.forward(self.ax,self.by)
        self.axpbypc=self.addg1.forward(self.axpby,c)
        return self.axpbypc

    def backward(self,gradient_top):
        self.axpbypc.gradient=gradient_top
        self.addg1.backward()
        self.addg0.backward()
        self.mulg1.backward()
        self.mulg0.backward()

class SVM(object):
    def __init__(self):
        self.a=Unit(1.0)
        self.b=Unit(-2.0)
        self.c=Unit(-1.0)
        self.circuit=Circuit()
    def forward(self,x,y):


        self.unit_out=self.circuit.forward(x,y,self.a,self.b,self.c)
        return self.unit_out

    def backward(self,label):
        self.a.gradient=0
        self.b.gradient=0
        self.c.gradient=0
        pull=0.0

        if label==1 and self.unit_out.value<1.:
            pull=1.
        if label==-1 and self.unit_out.value>-1.:
            pull=-1.

        # print 'label {} current output {} pull {}'.format(label,self.unit_out.value,pull)
        self.circuit.backward(pull)
        #regularization
        self.a.gradient+=-self.a.value
        self.b.gradient+=-self.b.value

    def fit(self,xtrain,ytrain,iter=400,stepsize=.01,verbose=False):
        self.xtrain=xtrain
        self.ytrain=ytrain
        for i in xrange(iter):
            ind=random.randint(0,len(xtrain)-1)
            x=Unit(xtrain[ind,0])
            y=Unit(xtrain[ind,1])
            label=ytrain[ind]
            self.learnfrom(x,y,label,stepsize)
            if i%25==0 and verbose==True:
                print 'iteration {} accuracy = {}'.format(i,self.score())



            #    print 'iteration {} training accuracy {}'.format(i,evalTrainingAccuracy(self,xtrain,ytrain))
    def predict(self,xtrain):
        y_predict=[]

        for i in xrange(len(xtrain)):
            x = Unit(xtrain[i, 0])
            y = Unit(xtrain[i, 1])

            if self.forward(x,y).value>0:
                y_predict.append(1)
            else:
                y_predict.append(-1)

        return np.array(y_predict)

    def score(self):

         return  np.sum(self.predict(self.xtrain)==self.ytrain)*1./len(self.ytrain)



    def learnfrom(self,x,y,label,stepsize):
        self.forward(x,y)
        #print x,y
        # print label
        # print x.value,y.value,self.a.value,self.b.value,self.c.value
        # print x.value*self.a.value+y.value*self.b.value+self.c.value,self.forward(x,y).value

        self.backward(label)


        self.parameterUpdate(stepsize)


    def parameterUpdate(self,stepsize):
        self.a.value+=stepsize*self.a.gradient
        self.b.value+=stepsize*self.b.gradient
        self.c.value+=stepsize*self.c.gradient






def evalTrainingAccuracy(clf,xtrain,ytrain):
    num_correct=0
    for i in xrange(len(xtrain)):
        x=Unit(xtrain[i,0])
        y=Unit(xtrain[i,1])
        label=ytrain[i]

        output=clf.forward(x,y)
        #print output
        if output>0:
            predict=1
        else:
            predict=-1


        if predict==label:
            num_correct+=1

    return num_correct*1./len(xtrain)








ytrain=[1,-1,1,-1,-1,+1]

xtrain=np.array(
[[1.2, 0.7],
[-0.3, 0.5],
[-3., -1.] ,
[0.1, 1.0],
[3.0, 1.1],
[2.1, -3]] )




svm=SVM()

svm.fit(xtrain,ytrain,iter=1000,stepsize=.5,verbose=True)






