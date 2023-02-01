#torch
def data_i():
  return [[0., 0.], [1., 0.], [0., 1.]]

def data_j():
  return [[0., 1.], [0., 1.], [1., 1.]]
#end.

inp(i) ~ data(@data_i).
inp(j) ~ data(@data_j).

0.5::a.
0.25::b :- a.

?::f(X) as @test_mlp on "hub" :- inp(X), a.

#query(f(i)).
#query(f(j)).
#query(f(i) | a).
#query(f(j) | b).
