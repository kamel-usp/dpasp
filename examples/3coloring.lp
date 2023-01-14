% Build a random graph with n vertices.
#const n = 5.
v(1..n).
% The choice of p reflects the sparsity/density of the random graph.
% A small p produces sparser graphs, while a large p prefers denser graphs.
0.5+::e(X, Y) :- v(X), v(Y), X < Y.
e(X, Y) :- e(Y, X).
% A color (here the predicate c/2) defines a coloring of a vertex.
% The next three lines define the uniqueness of a vertex's color.
c(X, r) :- not c(X, g), not c(X, b), v(X).
c(X, g) :- not c(X, r), not c(X, b), v(X).
c(X, b) :- not c(X, r), not c(X, g), v(X).
% Produce a contradiction if two neighbors have the same color.
f :- not f, e(X, Y), c(X, Z), c(Y, Z).

% Query the probability of vertex a being red.
#query(c(1, r)).
% Query the probability of the edge e(1, 2) existing given that the graph is not 3-colorable.
#query(e(1, 2) | undef f).
% Query the probability of the graph not being 3-colorable.
#query(undef f).
