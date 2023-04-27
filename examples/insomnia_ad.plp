sleep(X) :- not work(X), not insomnia(X), person(X).
work(X) :- not sleep(X), person(X).

person(anna).
person(bill).

?::insomnia(anna); ?::insomnia(bill); ?::insomnia(noone).

#query(insomnia(anna)).
#query(insomnia(bill)).

#query(work(anna)).
#query(work(bill)).

#query(sleep(anna)).
#query(sleep(bill)).
