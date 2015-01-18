__author__ = 'Sergey Matyunin'

import theano
import theano.tensor as T


class TrainFunction(object):
    def __init__(self, u0, v0, rate, num_steps, **kwargs):
        self.rate = rate
        

        self.gu, self.gv = None, None
        self.E = None
        self.count = 0
        self.num_steps = num_steps
        

    def done(self):
        return self.count >= self.num_steps

    def get_energy(self):
        raise Exception("Non implemented")

    def get_function(self):
        if self.E is None:
            self.E = self.get_energy()

        if self.gu is None or self.gv is None:
            self.gu, self.gv = T.grad(self.E, [self.tu, self.tv])

        train_function = theano.function(
            inputs=[],
            outputs=[self.E],
            updates=((self.tu, self.tu - self.rate * self.gu), (self.tv, self.tv - self.rate * self.gv))
        )

        return train_function

    def init(self, u0, v0, Ix, Iy, It):
        self.count = 0
        self.tu = theano.shared(u0, name='tu', borrow=True)
        self.tv = theano.shared(v0, name='tv', borrow=True)
        self.tIx = theano.shared(Ix, name="tIx", borrow = True)
        self.tIy = theano.shared(Iy, name = "tIy", borrow = True)
        self.tIt = theano.shared(It, name="tIt", borrow = True)

        self.train_function = self.get_function()

    def step(self, *args):
        self.count += 1
        return self.train_function()


class TrainFunctionSimple(TrainFunction):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.get('alpha', 1.1)

        super(self.__class__, self).__init__(*args, **kwargs)

    def get_energy(self,):
        Edata = T.sum((self.tIx * self.tu + self.tIy * self.tv + self.tIt) ** 2)
        
        Ereg1 = T.sum(self.tu**2 + self.tv**2)
        return Edata+self.alpha*(Ereg1)   


class TrainFunctionTV(TrainFunction):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.get('alpha', 1.1)

        super(self.__class__, self).__init__(*args, **kwargs)

    def get_energy(self,):
        Edata = T.sum((self.tIx * self.tu + self.tIy * self.tv + self.tIt) ** 2)
        Ereg1 = T.sum(
            (self.tu[1:]-self.tu[:-1])**2 +
            (self.tv[1:]-self.tv[:-1])**2 )
        Ereg2 = T.sum(
            (self.tu[:,1:]-self.tu[:,:-1]) **2 +
            (self.tv[:,1:]-self.tv[:,:-1]) ** 2)

        return Edata+self.alpha*(Ereg1+Ereg2)   