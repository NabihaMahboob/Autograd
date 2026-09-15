import unittest
from engine import Value


class TestValue(unittest.TestCase):

    def test_square(self):
        x = Value(3.0)
        y = x ** 2
        y.backward()

        print(f"x²: expected grad = 6.0, actual grad = {x.grad}")
        self.assertAlmostEqual(x.grad, 6.0)

    def test_multiple_paths(self):
        x = Value(3.0)
        y = x * x + x
        y.backward()

        print(f"Multiple paths: expected grad = 7.0, actual grad = {x.grad}")
        self.assertAlmostEqual(x.grad, 7.0)

    def test_multiplication(self):
        x = Value(2.0)
        y = Value(4.0)
        z = x * y
        z.backward()

        print(f"Multiplication dx: expected grad = 4.0, actual grad = {x.grad}")
        print(f"Multiplication dy: expected = 2.0, actual = {y.grad}")
        self.assertAlmostEqual(x.grad, 4.0)
        self.assertAlmostEqual(y.grad, 2.0)

    def test_chain_rule(self):
        x = Value(2.0)
        y = x ** 2
        z = y ** 3
        z.backward()

        print(f"Chain rule: expected grad = 192.0, actual grad = {x.grad}")
        self.assertAlmostEqual(x.grad, 192.0)

    def test_tanh(self):
        x = Value(0.0)
        y = x.tanh()
        y.backward()

        print(f"Tanh: expected grad = 1.0, actual grad = {x.grad}")
        self.assertAlmostEqual(x.grad, 1.0)


if __name__ == "__main__":
    unittest.main()