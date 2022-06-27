% The Three Prisoners Problem, as shown in Example 3 of On the Complexity of Propositional and
% Relational Credal Networks. Fabio Gagliardi Cozman and Denis Deratani Mauá. International Journal
% of Approximate Reasoning. 2017.

0.5::a. 0.5::b. 0.5::c.

e1 :- a, not b, not c.
e2 :- not a, b, not c.
e3 :- not a, not b, c.

u :- e1.
u :- e2.
u :- e3.

#query(e1 | u)
