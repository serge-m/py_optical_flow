__author__ = 'Sergey Matyunin'

import theano
import theano.tensor as T


class TrainFunction(object):
    def __init__(self, u0, v0, rate, num_steps):
        self.rate = rate
        self.tu = theano.shared(u0, name='tu')
        self.tv = theano.shared(v0, name='tv')
        self.tIx = T.matrix("tIx")
        self.tIy = T.matrix("tIy")
        self.tIt = T.matrix("tIt")

        self.gu, self.gv = None, None
        self.E = None
        self.count = 0
        self.num_steps = num_steps
        self.train_function = self.get_function()

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
            inputs=[self.tIx, self.tIy, self.tIt],
            outputs=[self.E],
            updates=((self.tu, self.tu - self.rate * self.gu), (self.tv, self.tv - self.rate * self.gv)),
            allow_input_downcast=True)

        return train_function

    def init(self, u0, v0):
        self.count = 0
        self.tu.set_value(u0)
        self.tv.set_value(v0)

    def step(self, *args):
        self.count += 1
        return self.train_function(*args)


class TrainFunctionSimple(TrainFunction):
    def __init__(self, *args, **kwargs):
        self.alpha = kwargs.get('alpha', 1.1)

        super(self.__class__, self).__init__(*args, **kwargs)

    def get_energy(self,):
        Edata = T.sum((self.tIx * self.tu + self.tIy * self.tv + self.tIt) ** 2)
#         Ereg1 = T.sum(
#             (self.tu[1:]-self.tu[:-1])**2 +
#             (self.tv[1:]-self.tv[:-1])**2 )
#         Ereg2 = T.sum(
#             (self.tu[:,1:]-self.tu[:,:-1]) **2 +
#             (self.tv[:,1:]-self.tv[:,:-1]) ** 2)

        Ereg1 = T.sum(self.tu**2 + self.tv**2)
        Ereg2 = 0
        #+alpha*(Ereg1+Ereg2)
        return Edata+self.alpha*(Ereg1+Ereg2)