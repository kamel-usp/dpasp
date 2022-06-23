% Example 5 of On the Semantics and Complexity of Probabilistic Logic Programs. Fabio Cozman and
% Denis Mauá. Journal of Artificial Intelligence Research. 2017.

0.7::burglary. 0.2::earthquake.
0.9::alarm :- burglary, earthquake.
0.8::alarm :- burglary, not earthquake.
0.1::alarm :- not burglary, earthquake.
calls(X) :- alarm, neighbor(X).
neighbor(a). neighbor(b).

#query(alarm | burglary; earthquake)
#query(alarm | not burglary; earthquake)
#query(alarm | burglary; not earthquake)
#query(alarm | not burglary; not earthquake)
