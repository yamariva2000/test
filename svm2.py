import random
import numpy as np
ytrain=[1,-1,1,-1,-1,+1]

xtrain=np.array(
[[1.2, 0.7],
[-0.3, 0.5],
[-3., -1.] ,
[0.1, 1.0],
[3.0, 1.1],
[2.1, -3]] )



a,b,c = 1,-2,-1 #// initial parameters
for iter in xrange(4000):
  # pick a random data point
  i = random.randint(0,len(ytrain)-1)
  x = xtrain[i][0]
  y = xtrain[i][1]
  label = ytrain[i]

  #// compute pull
  score = a*x + b*y + c
  pull = 0.0
  if label == 1 and score < 1: pull = 1

  if label ==-1 and score > -1:pull = -1

  #// compute gradient and update parameters
  step_size = 0.01;
  a += step_size * (x * pull - a) #-a is from the regularization
  b += step_size * (y * pull - b)# ; // -b is from the regularization
  c += step_size * (1 * pull)


predict=[]
for i in xrange(len(xtrain)):
    p=a*xtrain[i,0]+b*xtrain[i,1]+c
    if p>0: predict.append(1)
    if p<0: predict.append(-1)

predict=np.array(predict)

print predict==ytrain
