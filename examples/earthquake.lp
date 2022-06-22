0.7::burglary. 0.2::earthquake.
0.9::alarm :- burglary, earthquake.
0.8::alarm :- burglary, not earthquake.
0.1::alarm :- not burglary, earthquake.
calls(X) :- alarm, neighbor(X).
neighbor(a). neighbor(b).

#query(calls(a))
#query(calls(a) | earthquake)
