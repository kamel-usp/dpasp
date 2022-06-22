% The Barber Paradox.

shaves(X, Y) :- barber(X), villager(Y), not shaves(Y, Y).
villager(a). barber(b). 0.5::villager(b).

% When villager(b) is a fact, the PLP fails to have a credal semantics. Query results are not very
% meaningful in this case.
#query(shaves(b, a))
#query(-shaves(b, b))
