0.7::burglary.
0.05::earthquake(heavy); 0.15::earthquake(mild); 0.8::earthquake(none).

0.90::alarm :- burglary, earthquake(heavy).
0.85::alarm :- burglary, earthquake(mild).
0.80::alarm :- burglary, earthquake(none).
0.10::alarm :- not burglary, earthquake(mild).
0.30::alarm :- not burglary, earthquake(heavy).

0.8::calls(X) :- alarm, neighbor(X).
0.1::calls(X) :- not alarm, neighbor(X).

neighbor(a). neighbor(b).

#query(alarm | burglary, earthquake(heavy)).
#query(alarm | not burglary, earthquake(mild)).
#query(alarm | burglary, not earthquake(mild)).
#query(alarm | not burglary, earthquake(none)).
