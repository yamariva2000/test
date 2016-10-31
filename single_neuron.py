



class Unit(object):

    def __init__(self,value,gradient=0):
        self.value=value
        self.gradient=gradient
    def __str__(self):        return  'value {}  gradient {} '.format(self.value,self.gradient)


class multiplyGate(object):

    def forward(self,u0,u1):
        self.u0=u0
        self.u1=u1
        self.utop=Unit(self.u0.value*self.u1.value)
        return self.utop

    def backward(self):
        self.u0.gradient=self.u1.value*self.utop.gradient
        self.u1.gradient=self.u0.value*self.utop.gradient

    def __str__(self):
        return 'u0.value={}  u0.grad={} u1.value={} u1.grad={} output={}'.format(self.u0.value,self.u0.gradient,self.u1.value,self.u1.gradient,self.utop)



a=Unit(1.0)
x=Unit(5.0)


neuron=multiplyGate()

output=neuron.forward(a,x)

print neuron

output.gradient=1

neuron.backward()

print neuron

step=.0001
a.value+=step*a.gradient
x.value+=step*x.gradient

output=neuron.forward(a,x)

print neuron


h=1


print ((1+h)*5-(1*5))/h
print (1*(5+h)-1*5)/h


