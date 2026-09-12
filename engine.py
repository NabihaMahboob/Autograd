class Value:
    def __init__(self, data, _children=(), operation=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self.operation = operation
        self.backward = lambda: None

    def __add__(self, other):
        output = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output.backward = _backward
        return output
    

        

    def __mul__(self, other):
        output = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += output.grad * other.data
            other.grad += output.grad * self.data

        output.backward = _backward
        return output
        

    def __repr__(self):
        return f"{self.data}"
    

a = Value(2.0)
b = Value(3.0)

c = a + b
d = a * b

print(c)   
print(d)
print(c._prev)
print(c.operation)

e = (a + b) * b

print(e)
print(e.operation)
print(e._prev)