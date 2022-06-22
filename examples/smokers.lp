% Example 6 of Inference and Learning in Probabilistic Logic Programs Using Weighted Boolean
% Formulas. Fierens et al. Theory and Practice of Logic Programming. 2014.

smokes(X) :- stress(X).
smokes(X) :- influences(Y, X), smokes(Y).
0.3::influences(a, b). 0.3::influences(b, a). 0.2::stress(b).

#query(smokes(a))
#query(smokes(b))
