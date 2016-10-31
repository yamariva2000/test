import  numpy  as  np
import  time  as  t
x  = np . random . randn(10e6 ) . astype (np . float32 )
start  =  t . time ( )
valid  = np . logical_and(-1<x,x<1)


print "CPU: Found %d values in %f secs" % (np.sum( valid) , t.time()-start)





import  gnumpy  as  g
x_gpu  = g .garray ( x ) .reshape(
