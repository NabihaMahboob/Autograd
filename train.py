from network import MLP
from engine import Value


def train(model, xs, ys, learning_rate=0.001, target_loss=0.01, max_epochs=1000):
    for epoch in range(max_epochs):
        ypred = [model(x) for x in xs]
        loss = sum((ygt-yout)**2 for ygt, yout in zip(ys, ypred))
        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data += learning_rate * -p.grad

        print(epoch, loss.data)
        if loss.data <= target_loss:
            break

    ypred = [model(x) for x in xs]
    loss = sum((ygt - yout) ** 2 for ygt, yout in zip(ys, ypred))
    return ypred, loss.data


model = MLP(3, [4, 4, 1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]
yb = [0.9, -0.8, -0.9, 0.8]

predictions, loss = train(model, xs, yb)
print(predictions)
print(loss)
