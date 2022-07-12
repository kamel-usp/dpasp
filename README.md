# PASP -- Credal Semantics for Probabilistic Answer Set Programming

Prototype implementation of the credal semantics for probabilistic ASP [[1]](#ref-1)[[2]](#ref-2).
Example probabilistic logic programs may be found in [`examples/`](examples/). Let's take a look at
some examples to show how to do inference with this package.

## Examples

Let's first take a look at the popular `asia` Bayesian network, here encoded as the probabilistic
logic program [`examples/asia.lp`](examples/asia.lp) in a
[ProbLog](https://dtai.cs.kuleuven.be/problog/) inspired syntax.

```clingo
0.01::trip. 0.5::smoking.
0.05::tuberculosis :- trip.
0.01::tuberculosis :- not trip.
0.1::cancer :- smoking.
0.01::cancer :- not smoking.
or :- tuberculosis. or :- cancer.
0.98::test :- or.
0.05::test :- not or.
```

The code below specifies the probabilities and the rules of our domain. Any fact can be preceded by
a probability `p` followed by double colons `::` (e.g., `0.01::trip` and `0.5::smoking`), meaning
that this fact may appear in the logic program with probability `p` and not appear with probability
`1-p`; we call these *probabilistic facts*. If no probability is given, then the fact must always
appear.

Besides facts, we also have rules. These are of the form `head :- body`, implications stating that
`head` will be proven true if `body` is provably true. For now, `head` must either be empty (an
integrity constraint) or a single atom. On the other hand, `body` may contain multiple atoms,
called *subgoals*. Similar to facts, we may have some probability coupled with rules, in which case
we call them *probabilistic rules*. Probabilistic rules follow the same syntax as probabilistic
facts; in fact, they are only syntax sugar, as the package unrolls them as a rule with a unique
probabilistic fact as one of its subgoals.

```clingo
% The following probabilistic rule
0.05::tuberculosis :- trip.

% is equivalent to
0.05::a.
tuberculosis :- trip, a.

% where a is a unique atom.
```

Let's take a look at the remaining lines in [`examples/asia.lp`](examples/asia.lp). To query for
probabilities, we use a similar syntax to [PASOCS](https://arxiv.org/abs/2105.10908).

```clingo
#query(trip)
#query(tuberculosis | trip)
#query(cancer | smoking)
#query(test | or)
#query(smoking)
#query(tuberculosis | not trip)
#query(cancer | not smoking)
#query(test | not or)
```

The lines above ask for the probabilities of atoms. For instance, `#query(trip)` is asking for the
probability that `trip` is provably true $\mathbb{P}(\texttt{trip})$, while `#query(tuberculosis |
not trip)` is asking for the probability of `tuberculosis` being provably true given that we know
`trip` not to be $\mathbb{P}(\texttt{tuberculosis}|\neg\texttt{trip})$. We may ask more complex
queries by aggregating atoms with a comma; for instance `#query(tuberculosis, not cancer | not
smoking, trip)` asks the probability
$\mathbb{P}(\texttt{tuberculosis},\neg\texttt{cancer}|\neg\texttt{smoking},\texttt{trip})$.

Since we are working with the credal semantics, each one of these queries will return a tuple of
lower and upper probabilities. Let's ask the package to provide these probabilities. We first
inspect our Probabilistic Logic Program (PLP).

```python
import pasp

P = pasp.parse("examples/asia.lp")
print(P)
```
```
<tuberculosis :- trip, __unique_id_1.
tuberculosis :- not trip, __unique_id_2.
cancer :- smoking, __unique_id_3.
cancer :- not smoking, __unique_id_4.
or :- tuberculosis. or :- cancer.
test :- or, __unique_id_5.
test :- not or, __unique_id_6.,
[0.01::trip., 0.5::smoking., 0.05::__unique_id_1., 0.01::__unique_id_2., 0.1::__unique_id_3., 0.01::__unique_id_4., 0.98::__unique_id_5., 0.05::__unique_id_6.],
[ℙ(trip), ℙ(tuberculosis | trip), ℙ(cancer | smoking), ℙ(test | or), ℙ(smoking), ℙ(tuberculosis | not trip), ℙ(cancer | not smoking), ℙ(test | not or)]>
```

A PLP, here the Python object `P`, is a triple $\langle L,PF,Q \rangle$, where $L$ is the logic program composed
solely of logic facts and rules, $PF$ are the probabilistic facts, and $Q$ are the queries to be
asked from the solver. We can see from the output above the generated rules and probabilistic facts
produced by the unrolling of probabilistic rules as well as the queries to be asked.

Let's ask the solver to produce the probabilities we asked exactly. We can do so by running the
`exact` function. Note that running exact inference is costly. Approximate inference is planned for
this package.

```python
R = pasp.exact(P)
```
```
ℙ(trip) = [0.010000000000000009, 0.010000000000000009]
ℙ(tuberculosis | trip) = [0.04999999999999998, 0.04999999999999998]
ℙ(cancer | smoking) = [0.1, 0.1]
ℙ(test | or) = [0.98, 0.98]
ℙ(smoking) = [0.5000000000000001, 0.5000000000000001]
ℙ(tuberculosis | not trip) = [0.010000000000000002, 0.010000000000000002]
ℙ(cancer | not smoking) = [0.009999999999999995, 0.009999999999999995]
ℙ(test | not or) = [0.04999999999999998, 0.04999999999999998]
```

Function `pasp.exact` returns the results of the queries as a list of pairs of lower and upper
probabilities in the order the queries are asked for in the PLP code.

Since [`examples/asia.lp`](examples/asia.lp) comes from a Bayesian network and therefore is an
acyclic PLP, the probabilities returned are sharp. Let's take a look at another (very simple)
example where this is not the case: [`examples/insomnia.lp`](examples/insomnia.lp).

```clingo
sleep :- not work, not insomnia. work :- not sleep.
0.3::insomnia.

#query(insomnia)
#query(work)
#query(sleep)
#query(not sleep)
```

Note that `sleep` and `work` produce an even loop when `insomnia` is set to false in the program,
essentially resulting in two possible stable models: one where only `work` is set to true and the
other where only `sleep` is true, each having different sets of probabilities. Let's query!

```python
pasp.exact(pasp.parse("examples/insomnia.lp"))
```
```
ℙ(insomnia) = [0.3, 0.3]
ℙ(work) = [0.3, 1.0]
ℙ(sleep) = [0.0, 0.7]
ℙ(not sleep) = [0.3, 1.0]
```

We now have the right lower and upper probabilities taking into account all possible stable models
of the PLP. This shows us that the probability of `sleep`, for instance, can take values low as
`0.0` and high as `0.7`, while `work` has at least `0.3` mass.

## Usage

For now, `pasp` is only to be run locally. Clone this repository to a directory of your choice, say
`pasp`. Change your working directory to it and then simply run

```python
import pasp
```

to have access to the exported symbols in the package.

## References

<div id="ref-1">[1] - The Joy of Probabilistic Answer Set Programming: Semantics, Complexity, Expressivity,
Inference. Fabio Gagliardi Cozman and Denis Deratani Mauá. In International Journal of Approximate
Reasoning 125. 2020.</div>
<br>

<div id="ref-2">[2] - On the Semantics and Complexity of Probabilistic Logic Programs. Fabio Gagliardi Cozman and
Denis Deratani Mauá. In Journal of Artificial Intelligence Research 60. 2017.</div>

