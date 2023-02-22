% This is a minimal example of neural ADs in pasp.

#torch
def train():
  # Input of the neural network during training.
  return [*[[0., 0.]]*8, *[[0., 1.]]*4, *[[1., 0.]]*5, *[[1., 1.]]*3]

def test():
  # Network inputs for evaluation.
  return [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
#end.

% Just like in neural rules, predicate g(x) defines the data (train and test datasets) to be used
% in the network.
g(x) ~ test(@test), train(@train).

% The neural network is defined in function test_ad_mlp in hub/hubconf.py as a PyTorch Hub model.
?::f(X, {a, b, c}) as @test_ad_mlp on "hub" with lr = 0.1, weight_decay = 0.001, momentum = 0.5 :- g(X).

% We want to evaluate the outputs of the network f (here characterized by possible values a, b, c).
#query(f(x, a)).
#query(f(x, b)).
#query(f(x, c)).

% The program above, when run with the Python script below
%
%   P = pasp.parse("examples/neural_ad_minimal.lp")
%   print("Initial probability values:")
%   pasp.exact(P, psemantics = "maxent")
%   D = [["f(x, a)"], ["f(x, a)"], ["f(x, b)"], ["f(x, b)"], ["f(x, b)"], ["f(x, b)"], ["f(x, b)"],
%        ["f(x, c)"], ["f(x, a)"], ["f(x, b)"], ["f(x, b)"], ["f(x, c)"], ["f(x, a)"], ["f(x, b)"],
%        ["f(x, b)"], ["f(x, c)"], ["f(x, c)"], ["f(x, a)"], ["f(x, b)"], ["f(x, c)"]]
%   pasp.learn(P, D, niters = 1000, alg = "lagrange")
%   print("Post-learning probabilities:")
%   pasp.exact(P, psemantics = "maxent")
%
% produces the following output.
%
%   Initial probability values:
%   ℙ(f(x,a)) = 0.300000
%   ℙ(f(x,b)) = 0.200000
%   ℙ(f(x,c)) = 0.500000
%   ---
%   ℙ(f(x,a)) = 0.500000
%   ℙ(f(x,b)) = 0.100000
%   ℙ(f(x,c)) = 0.400000
%   ---
%   ℙ(f(x,a)) = 0.600000
%   ℙ(f(x,b)) = 0.300000
%   ℙ(f(x,c)) = 0.100000
%   ---
%   ℙ(f(x,a)) = 0.999946
%   ℙ(f(x,b)) = 0.000018
%   ℙ(f(x,c)) = 0.000036
%   ---
%   Post-learning probabilities:
%   ℙ(f(x,a)) = 0.247704
%   ℙ(f(x,b)) = 0.626366
%   ℙ(f(x,c)) = 0.125929
%   ---
%   ℙ(f(x,a)) = 0.255993
%   ℙ(f(x,b)) = 0.495856
%   ℙ(f(x,c)) = 0.248152
%   ---
%   ℙ(f(x,a)) = 0.200541
%   ℙ(f(x,b)) = 0.399840
%   ℙ(f(x,c)) = 0.399619
%   ---
%   ℙ(f(x,a)) = 0.330774
%   ℙ(f(x,b)) = 0.335449
%   ℙ(f(x,c)) = 0.333777
%   ---
%
% which is close to the ground truth probabilities:
%
%   [[2/8, 5/8, 1/8], [1/4, 2/4, 1/4], [1/5, 2/5, 2/5], [1/3, 1/3, 1/3]]
%     =
%   [[0.25, 0.625, 0.125], [0.25, 0.5, 0.25], [0.2, 0.4, 0.4], [0.3333, 0.3333, 0.3333]]
