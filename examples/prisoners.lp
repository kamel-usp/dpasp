% The Three Prisoners Problem, as shown in Example 3 of On the Complexity of Propositional and
% Relational Credal Networks. Fabio Gagliardi Cozman and Denis Deratani Mauá. International Journal
% of Approximate Reasoning. 2017.

% Prisoners A, B and C are in prison waiting for their execution, to happen tomorrow. Tonight the
% governor can pardon each one of them; the decision to pardon each prisoner is taken by throwing
% three coins independently. If nothing is assumed from the coins fairness, then their probability,
% here given by the credal facts a, b and c, can take any value in [0, 1], which doesn't give us
% much information down the line. Instead, let's assume a somewhat biased coin, possibly taking any
% value from the credal set [α, 1-α], where α = 19/40.
[0.475, 0.525]::a. [0.475, 0.525]::b. [0.475, 0.525]::c.

% During the night, the honest guard learns the decisions made by the governor. The guard and
% prisoner A are talking; the guard says that one and only one prisoner will be executed, and the
% others will be released. This is represented by the (deterministic) rule u, which is equivalent
% to the disjunction of the rules e1, e2 and e3, indicating if a, b or c is to be executed,
% respectively.
e1 :- a, not b, not c.
e2 :- not a, b, not c.
e3 :- not a, not b, c.
u :- e1. u :- e2. u :- e3.

% Prisoner A then concludes that the probability that they are the one being executed must be in
% the interval ℙ(e1 | u) ≈ [0.2904, 0.3792]. The query below asks exactly that.
#query(e1 | u)

% During the conversation with the guard, Prisoner A asks the guard to tell him the name of a
% prisoner, other than himself, who will be released (certainly at least one of the other prisoners
% will be released). The guard says B. Thinking a bit about it, A gets desperate: before asking,
% his (unconditional) probability of execution was in the credal set [0.2904, 0.3792], but now
% it seems that, just by asking, his (conditional) credal set of execution became ≈ [0.4501, 0.5498],
% i.e. much more likely to be executed. Does this conclusion make sense?
#query(e1 | not b, u)

% Denote by g the proposition "guard says to A that prisoner B, will be released". To obtain ℙ(e1 |
% g ∧ u), we do have to model the guard's response. There are two situations that cause the guard
% to say that B will be released. If B will be released, and C will be executed, then the guard
% must say B. If instead both B and C will be released, then the guard can say B or C. One way to
% express this reasoning is to introduce rules
[0, 1]::d.
c1 :- not e1, e3.
c2 :- not e2, not e3, d.
g :- c1. g :- c2.
% where d captures the choice of the guard towards B when the guard can say B or C. If we assume
% this to be anything (i.e., in the credal set [0, 1]) then it must be then that ℙ(g | e1 ∧ u) =
% ℙ(d) and ℙ(e1 | g ∧ u) = [0, (1+(α/(1-α))²)⁻¹], where ℙ⁻ is the upper probability. As previously,
% assuming α = 19/40 gives us ℙ(e1 | g ∧ u) ≈ [0, 0.5499]. Let's validate this programatically:
#query(g | e1, u)
#query(d)
#query(e1 | g, u)

% Finally, suppose alternatively that the guard decides by flipping a coin ℙ(d) ∈ [α, 1-α], just
% like in the case of the governor's. We then have ℙ(e1 | g ∧ u) = [(1+((1-α)/a)²⋅α⁻¹)⁻¹,
% (1+(α/(1-α))²⋅(1-α)⁻¹)⁻¹] ≈ [0.2800, 0.3907]:
[0.475, 0.525]::da.
c2a :- not e2, not e3, da.
ga :- c1. ga :- c2a.
#query(e1 | ga, u)
