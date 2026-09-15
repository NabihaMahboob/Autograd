import random

from engine import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for i in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act=0
        for num, weight in zip(x, self.w):
            act+=num*weight
        act+=self.b
        return act.tanh()

n = Neuron(3)
x = [2.0, 3.0, -1.0]

out = n(x)

print(out)
out.backward()

print(n.w[0].grad)
print(n.w[1].grad)
print(n.w[2].grad)
print(n.b.grad)