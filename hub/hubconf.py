import torch

dependencies = ["torch"]

def test_mlp():
  """ Test MLP for neural rule testing.

  Network encoding the following discrete distribution:

    +---+---+---------+
    | a | b | P(a, b) |
    +---+---+---------+
    | 0 | 0 |   0.3   |
    | 0 | 1 |   0.5   |
    | 1 | 0 |   0.6   |
    | 1 | 1 |   1.0   |
    +---+---+---------+
  """
  class Net(torch.nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.f = torch.nn.Linear(2, 2)
      self.g = torch.nn.Sigmoid()
      self.h = torch.nn.Linear(2, 1)

    def forward(self, x: torch.Tensor):
      return self.h(self.g(self.f(x)))

  N = Net()
  S = {
    "f.weight": torch.tensor([[ 1.1519355, 0.9704726], [-1.3226409, -0.77811617]]),
    "f.bias":   torch.tensor([-2.534905, 2.3281014]),
    "h.weight": torch.tensor([[1.4467078, -0.6486825]]),
    "h.bias":   torch.tensor([0.7848085])
  }
  N.load_state_dict(S)
  return N

def test_ad_mlp():
  """ Test MLP for neural AD testing.

  Suppose we have an annotated disjunction

    P(a)::f(X, a); P(b)::f(X, b); P(c)::f(X, c).

  The network encodes the following discrete distribution:

    +-----+-----+------+------+------+
    | X_0 | X_1 | P(a) | P(b) | P(c) |
    +-----+-----+------+------+------+
    |  0  |  0  |  .3  |  .2  |  .5  |
    |  0  |  1  |  .5  |  .1  |  .4  |
    |  1  |  0  |  .6  |  .3  |  .1  |
    |  1  |  1  |  1.  |  .0  |  .0  |
    +---+---+---+------+------+------+
  """
  class Net(torch.nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.f = torch.nn.Linear(2, 2)
      self.g = torch.nn.Sigmoid()
      self.h = torch.nn.Linear(2, 3)

    def forward(self, x: torch.Tensor):
      return self.h(self.g(self.f(x)))

  N = Net()
  S = {
    "f.weight": torch.tensor([[1.4857, 0.3382], [1.6301, 1.9893]]),
    "f.bias":   torch.tensor([-0.6389, -3.2496]),
    "h.weight": torch.tensor([[ 0.5377,  0.8554],
                              [ 0.5681, -0.7919],
                              [-1.1057, -0.0635]]),
    "h.bias":   torch.tensor([0.0823, 0.0333, 0.8844])
  }
  N.load_state_dict(S)
  return N
