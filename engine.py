import math

class Value:
    def __init__(self, data, _children=(), operation=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._operation = operation
        self._backward = lambda: None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        output = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        output = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += output.grad * other.data
            other.grad += output.grad * self.data

        output._backward = _backward
        return output

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        output = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += output.grad * other * self.data ** (other-1)

        output._backward = _backward
        return output

    def tanh(self):
        output = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += output.grad * (1-output.data**2)

        output._backward = _backward
        return output

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1)
        
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return (-self) + other

    def __rtruediv__(self, other):
        return other * (self**-1)

    def __neg__(self):
        return self * -1
    
    def __repr__(self):
        return f"{self.data}"


    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()
         