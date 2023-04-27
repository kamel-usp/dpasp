% Example 7.3 of The Well-Founded Semantics for General Logic Programs. Van Gelder et al. Journal
% of the Association for Computing Machinery. 1991.

wins(X) :- move(X, Y), not wins(Y).
move(a, b). move(b, a). move(b, c). 0.3::move(c, d).

#query(wins(b))
#query(wins(c))
