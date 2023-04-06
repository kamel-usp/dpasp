% This is the simplest case possible of neural rules in pasp.

#torch
def net():
  # Define the neural network as a single neuron with a sigmoid as activation function.
  N = torch.nn.Sequential(
    torch.nn.Linear(1, 1, bias = True),
    # Make sure the output is always in [0, 1].
    torch.nn.Sigmoid()
  )
  # Initialize parameters to uniform distribution.
  N.load_state_dict({"0.weight": torch.tensor([[0.0]]), "0.bias": torch.tensor([0.0])})
  return N

def train():
  # Input of the neural network during training.
  return [*[[0.]]*10, *[[1.]]*10]

def test():
  # Network inputs for evaluation.
  return [[0.], [1.]]
#end.

% Predicate g(x) is a special data predicate for declaring test and train datasets.
g(x) ~ test(@test), train(@train).
% Neural rule f(X) is defined by the function net, with optimizer parameters given after the with
% keyword. Neural components must always come accompanied of a data predicate (in this case, g(X)).
?::f(X) as @net with lr = 0.1, momentum = 0.1 :- g(X).

% For this case, f(X) is grounded to a single neural fact f(x). Let's evaluate its probability.
#query(f(x)).

% The program above, when run with the Python script below
%
%   P = pasp.parse("examples/neural_minimal.lp")
%   # Produces a dataset equivalent to P(f(x)) = 0.8 when the neural network input is 0.0,
%   # and P(f(x)) = 0.1 when the network's input is 1.0.
%   D = [*[["f(x)"]]*8, *[["~f(x)"]]*2, *[["f(x)"]]*1, *[["~f(x)"]]*9]
%   print("Initial probability values:")
%   pasp.exact(P, psemantics = "maxent")
%   # The fix-point learning algorithm does not support neural components.
%   pasp.learn(P, D, niters = 500, alg = "lagrange", eta = 1.)
%   print("Post-learning probabilities:")
%   pasp.exact(P, psemantics = "maxent")
%
% produces the following output.
%
%   Initial probability values:
%   ℙ(f(x)) = 0.500000
%   ---
%   ℙ(f(x)) = 0.500000
%   ---
%   Post-learning probabilities:
%   ℙ(f(x)) = 0.800000
%   ---
%   ℙ(f(x)) = 0.100000
%   ---
