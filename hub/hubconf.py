import torch

dependencies = ["torch"]

def test_mlp():
  """ Test MLP for neural rule testing.

  Network encoding the following conditional probability table on the probabilistic fact f:

    +---+---+-------------+
    | a | b | P(f | a, b) |
    +---+---+-------------+
    | 0 | 0 |     0.3     |
    | 0 | 1 |     0.5     |
    | 1 | 0 |     0.6     |
    | 1 | 1 |     1.0     |
    +---+---+-------------+
  """
  N = torch.nn.Sequential(
    torch.nn.Linear(2, 2),
    torch.nn.Sigmoid(),
    torch.nn.Linear(2, 1),
    # Make sure output is non-negative and at most 1.
    torch.nn.Sigmoid()
  )
  S = {
    "0.weight": torch.tensor([[5.404379367828369, 5.935782432556152],
                              [4.996634006500244, 4.26226806640625]]),
    "0.bias":   torch.tensor([-9.917333602905273, -5.738117694854736]),
    "2.weight": torch.tensor([[10.780776977539062, 3.557007312774658]]),
    "2.bias":   torch.tensor([-0.8592491745948792])
  }
  N.load_state_dict(S)
  return N

def test_ad_mlp():
  """ Test MLP for neural AD testing.

  Suppose we have an annotated disjunction

    P(a)::f(X, a); P(b)::f(X, b); P(c)::f(X, c).

  The network encodes the following conditional probability table on the annotated disjunction d:

    +-----+-----+----------+----------+----------+
    | X_0 | X_1 | P(d | a) | P(d | b) | P(d | c) |
    +-----+-----+----------+----------+----------+
    |  0  |  0  |    .3    |    .2    |    .5    |
    |  0  |  1  |    .5    |    .1    |    .4    |
    |  1  |  0  |    .6    |    .3    |    .1    |
    |  1  |  1  |    1.    |    .0    |    .0    |
    +-----+-----+----------+----------+----------+
  """
  N = torch.nn.Sequential(
    torch.nn.Linear(2, 4),
    torch.nn.Sigmoid(),
    torch.nn.Linear(4, 3),
    # Make sure output is non-negative.
    torch.nn.Softmax(dim = 1)
  )
  S = {
    "0.weight": torch.tensor([[-2.436516046524048, -1.9045624732971191],
                              [-3.4312784671783447, 7.422353744506836],
                              [3.002552032470703, 2.9324307441711426],
                              [-5.433633804321289, 1.480564832687378]]),
    "0.bias":   torch.tensor([3.18778133392334, 2.1292202472686768, -4.477348804473877, 0.029237013310194016]),
    "2.weight": torch.tensor([[-1.9041186571121216, 2.8515288829803467, 3.432497978210449, -2.0234572887420654],
                              [2.56845760345459, -2.964510679244995, -2.5871169567108154, 1.8192431926727295],
                              [1.7890348434448242, -0.4094463288784027, -3.4937069416046143, 2.4915430545806885]]),
    "2.bias":   torch.tensor([0.8296637535095215, -0.5550796985626221, -1.5046206712722778])
  }
  N.load_state_dict(S)
  return N
