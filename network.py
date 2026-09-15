import random

from engine import Value

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act=0
        for num, weight in zip(x, self.w):
            act+=num*weight
        act+=self.b
        return act.tanh()

    def __repr__(self):
        return f"Neuron({len(self.w)})"

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

    def parameters(self):
        params = []

        for neuron in self.neurons:
            params.extend(neuron.parameters())

        return params

class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0
    