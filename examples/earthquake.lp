0.7::burglary. 0.2::earthquake.
alarm :- burglary, earthquake, a1.
alarm :- burglary, not earthquake, a2.
alarm :- not burglary, earthquake, a3.
0.9::a1. 0.8::a2. 0.1::a3.
calls(X) :- alarm, neighbor(X).
neighbor(a). neighbor(b).
