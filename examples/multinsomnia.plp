sleep(X) :- not work(X), not insomnia(X), person(X).
work(X) :- not sleep(X), person(X).

calls(X, Y) :- not sleep(X), not sleep(Y), lonely(X), person(X), person(Y), X != Y.

person(anna).
person(bill).
person(charlie).

0.3::insomnia(anna).
0.5::insomnia(bill).
0.7::insomnia(charlie).

0.2::lonely(anna); 0.45::lonely(bill); 0.35::lonely(charlie).

#query(insomnia(anna)).
#query(insomnia(bill)).
#query(insomnia(charlie)).

#query(work(anna)).
#query(work(bill)).
#query(work(charlie)).

#query(sleep(anna)).
#query(sleep(bill)).
#query(sleep(charlie)).

#query(calls(anna, bill)).
#query(calls(anna, charlie)).
#query(calls(bill, anna)).
#query(calls(bill, charlie)).
#query(calls(charlie, anna)).
#query(calls(charlie, bill)).
