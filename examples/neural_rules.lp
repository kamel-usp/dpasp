#torch
def data_i():
  return [[0., 0.], [1., 0.], [0., 1.]]

def data_j():
  return [[0., 1.], [0., 1.], [1., 1.]]
#end.

inp(i) ~ test(@data_i), train(@data_i).
inp(j) ~ test(@data_j), train(@data_i).

0.5::a.
0.25::b :- a.

?::f(X) as @test_mlp on "hub" :- inp(X), a.

#query(f(i)).
#query(f(j)).
#query(f(i) | a).
#query(f(j) | b).
