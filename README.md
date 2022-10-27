# PASP - Probabilistic Answer Set Programming

[![Tests](https://github.com/RenatoGeh/pasp/actions/workflows/tests.yml/badge.svg)](https://github.com/RenatoGeh/pasp/actions/workflows/tests.yml)
[![Docs](https://github.com/RenatoGeh/pasp/actions/workflows/docs.yml/badge.svg)](https://github.com/RenatoGeh/pasp/actions/workflows/docs.yml)
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://renatogeh.github.io/pasp)
[![GitHub](https://img.shields.io/github/license/RenatoGeh/pasp?color=blue&label=License)](https://github.com/RenatoGeh/pasp/blob/master/LICENSE)

Prototype implementation for probabilistic ASP [[1]](#ref-1)[[2]](#ref-2). Example probabilistic
logic programs may be found in [`examples/`](examples/). See full API documentation
[here](https://renatogeh.github.io/pasp/pasp.html). Let's take a look at some examples to show
how to do inference with this package.

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

`pasp` also supports probabilistic rules with variables (e.g. `0.3::f(X) :- g(X).`), although these
are (for now) constrained to only normal rules. Internally, what `pasp` does is ground these rules
and then apply the transformation mentioned above, adding a probabilistic fact to the body of the
grounded rule. For efficiency reasons, subgoals in first-order probabilistic rules are only
grounded following the non-probabilistic part of the program; otherwise, at every total choice a
new grounded program would have to be generated.

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

If we are working with the credal semantics (more on this later), then each one of these queries
will return a tuple of lower and upper probabilities. Let's ask the package to provide these
probabilities (for the command-line syntax of `pasp`, see [Usage](#command-line-usage). We first
inspect our Probabilistic Logic Program (PLP).

```python
>>> import pasp
>>> P = pasp.parse("examples/asia.lp")
>>> print(P)
```
```
<Logic Program:
tuberculosis :- trip, __unique_id_4.
tuberculosis :- not trip, __unique_id_5.
cancer :- smoking, __unique_id_6.
cancer :- not smoking, __unique_id_7.
or :- tuberculosis.
or :- cancer.
test :- or, __unique_id_8.
test :- not or, __unique_id_9.,
Probabilistic Facts:
[0.01::trip, 0.5::smoking, 0.05::__unique_id_4, 0.01::__unique_id_5, 0.1::__unique_id_6, 0.01::__unique_id_7, 0.98::__unique_id_8, 0.05::__unique_id_9],
Credal Facts:
[]
Probabilistic Rules:
[0.05::tuberculosis :- trip, 0.01::tuberculosis :- not trip, 0.1::cancer :- smoking, 0.01::cancer :- not smoking, 0.98::test :- or, 0.05::test :- not or],
Queries
[ℙ(trip), ℙ(tuberculosis | trip), ℙ(cancer | smoking), ℙ(test | or), ℙ(smoking), ℙ(tuberculosis | not trip), ℙ(cancer | not smoking), ℙ(test | not or)]>
```

A PLP, here the Python object `P`, is a tuple $\langle L,PF,CF,PR,Q \rangle$, where $L$ is the
logic program composed solely of logic facts and rules, $PF$ are the probabilistic facts, $CF$ are
the credal facts (see [`examples/prisoners.lp`](examples/prisoners.lp)), $PR$ are probabilistic
rules, $and $Q$ are the queries to be asked from the solver. We can see from the output above the
generated rules and probabilistic facts produced by the unrolling of probabilistic rules as well as
the queries to be asked.

Let's ask the solver to produce the exact probabilities we asked. We can do so by running the
`exact` function. Note that running exact inference is costly. Approximate inference is planned for
this package.

```python
>>> R = pasp.exact(P)
```
```
ℙ(trip) = [0.010000, 0.010000]
ℙ(tuberculosis | trip) = [0.050000, 0.050000]
ℙ(cancer | smoking) = [0.100000, 0.100000]
ℙ(test | or) = [0.980000, 0.980000]
ℙ(smoking) = [0.500000, 0.500000]
ℙ(tuberculosis | not trip) = [0.010000, 0.010000]
ℙ(cancer | not smoking) = [0.010000, 0.010000]
ℙ(test | not or) = [0.050000, 0.050000]
```

Function `pasp.exact` returns the results of the queries as a tuple of pairs of lower and upper
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
#query(not work)
```

Note that `sleep` and `work` produce an even loop when `insomnia` is set to false in the program,
essentially resulting in two possible stable models: one where only `work` is set to true and the
other where only `sleep` is true, each having different sets of probabilities. Let's query!

```python
>>> pasp.exact(pasp.parse("examples/insomnia.lp"))
```
```
ℙ(insomnia) = [0.300000, 0.300000]
ℙ(work) = [0.300000, 1.000000]
ℙ(sleep) = [0.000000, 0.700000]
ℙ(not sleep) = [0.300000, 1.000000]
ℙ(not work) = [0.000000, 0.700000]
```

We now have the right lower and upper probabilities taking into account all possible stable models
of the PLP. This shows us that the probability of `sleep`, for instance, can take values low as
`0.0` and high as `0.7`, while `work` has at least `0.3` mass.

## Semantics

Probabilities and models may change depending on the *semantics* of the language. The semantics of
the language can be defined around two dimensions: the *logic semantics* and the *probabilistic
semantics*. We allow different logic and probabilistic semantics, and their different combinations,
in `pasp`. As of now, the following semantics are implemented:

#### Logic semantics

- Stable semantics;
- Partial semantics;
- L-Stable semantics.

#### Probabilistic semantics

- Credal semantics;
- MaxEnt semantics.

### Examples of logic semantics

Let us first examine the Barber Paradox example (see [`examples/barber.lp`](examples/barber.lp)).

```
shaves(X, Y) :- barber(X), villager(Y), not shaves(Y, Y).
villager(a). barber(b). 0.5::villager(b).
```

Under the stable semantics, we find ourselves in a pickle. With probability half, `villager(b)` is
chosen (or not) to appear in the program. Suppose it does not; then we know that when grounding the
program we get

```
shaves(b, a) :- barber(b), villager(a), not shaves(a, a).
```

which is fine, since `shaves(a, a)` is not in any model. Now suppose `villager(b)` is, in fact,
added to the program; then we have

```
shaves(b, b) :- barber(b), villager(b), not shaves(b, b).
```

in addition to the previous rule. This presents a problem, since `shaves(b, b) :- ..., not
shaves(b, b)` is a clear contradiction and so no stable model can be found in this total choice
under the stable model semantics. Since the credal semantics assumes the existence of a model for
any given total choice, the computed probabilities will be garbage (for now, ignore the `undef`
keyword):

```python
>>> P = pasp.parse("examples/barber.lp")
>>> pasp.exact(P)
```
```
ℙ(shaves(b,a)) = [1.000000, 0.500000]
ℙ(not shaves(b,b)) = [1.000000, 0.500000]
ℙ(undef shaves(b,b)) = [0.500000, 0.000000]
```

Instead of using the stable semantics, however, we could use the *partial semantics*. The partial
semantics attributes to every atom three possible values: true, false or undefined (represented by
the `undef` keyword). Atoms are set to undefined when its value could not be proved either true or
false, as is the case of contradictions.

Let us, again, analyze the Barber Paradox program, this time under the partial semantics. When
`villager(b).` is *not* in the program, then the partial semantics computes, in this case, the same
minimal model as in the stable semantics. However, when `villager(b)` is present in the program,
then we cannot attribute neither truth nor falsity to `shaves(b, b)`, and thus `shaves(b, b)` must
be set to undefined. Now, if we compute the credal probabilities again, this time under partial
semantics, we get

```python
>>> P = pasp.parse("examples/barber.lp", semantics = "partial")
>>> pasp.exact(P)
```
```
ℙ(shaves(b,a)) = [1.000000, 1.000000]
ℙ(not shaves(b,b)) = [0.500000, 0.500000]
ℙ(undef shaves(b,b)) = [0.500000, 0.500000]
```

which are correct: in both total choices, there is only a single model; the one with `villager(b)`
agrees with `not shaves(b, b)`, and the other (without `villager(b)`) sets it to undefined, meaning
that `shaves(b, b)` is either set to false or undefined with equal probability.

Another interesting semantic is the *L-stable semantics*, which in practice agrees with the stable
model semantics when there exist stable models and with the partial semantics otherwise. Let us
take a look at the 3-coloring graph problem (see [`examples/3coloring.lp`](examples/3coloring.lp)).

```
#const n = 5.
v(1..n).
0.5::e(X, Y) :- v(X), v(Y), X < Y.
e(X, Y) :- e(Y, X).
c(X, r) :- not c(X, g), not c(X, b), v(X).
c(X, g) :- not c(X, r), not c(X, b), v(X).
c(X, b) :- not c(X, r), not c(X, g), v(X).
f :- not f, e(X, Y), c(X, Z), c(Y, Z).
```

The program above models the 3-coloring problem of graphs with `n = 5` vertices. Each total choice
models a different graph based on the presence (or not) of edges `e(X, Y)`. Here, the rule

```
f :- not f, e(X, Y), c(X, Z), c(Y, Z).
```

defines whether or not the graph is 3-colorable. Notice the presence of contradiction `f :- not f,
...`. When no 3-coloring can be achieved, then `f` must be set to `undef`, as the subgoals would
then hold and thus cause the contradiction. Under the partial semantics, any minimal partial model
is valid, and thus, even if we do find a 3-colorable graph under a total choice, other partial
models with `undef` attributions would possibly appear as potential minimal models.

```python
>>> P = pasp.parse("examples/3coloring.lp", semantics = "partial")
>>> pasp.exact(P)
```
```
ℙ(c(1,r)) = [0.000000, 1.000000]
ℙ(e(1,2) | undef f) = [0.090747, 0.971537]
ℙ(undef f) = [0.064453, 0.999023]
```

Now, if we choose the L-stable semantics

```python
>>> P = pasp.parse("examples/3coloring.lp", semantics = "lstable")
>>> pasp.exact(P)
```
```
ℙ(c(1,r)) = [0.000000, 1.000000]
ℙ(e(1,2) | undef f) = [0.772727, 0.772727]
ℙ(undef f) = [0.064453, 0.064453]
```

we find that the probability of getting non-3-colorable graphs drastically falls, as we discard
minimal models with `undef` when stable models are found.

### Examples of probabilistic semantics

Consider the game example shown in [`examples/game.lp`](examples/game.lp).

```
wins(X) :- move(X, Y), not wins(Y).
move(a, b). move(b, a). move(b, c). 0.3::move(c, d).
```

If we were to use the credal semantic, we would find that `wins(b)` appears in two of the stable
models (one when `move(c, d)` is in the program, and the other when it is not), and so by computing
the total choice probabilities, it either appears with probability 1.0-0.3 or 1.0, producing the
credal result `[0.7, 1.0]` below.

```python
>>> P = pasp.parse("examples/game.lp")
>>> pasp.exact(P)
```
```
ℙ(wins(b)) = [0.700000, 1.000000]
ℙ(wins(c)) = [0.300000, 0.300000]
```

When querying under the credal semantics, we account for the lower and upper probabilities of each
total choice and evaluate the final credal probabilities accordingly. If we wish to, instead,
uniformly consider models, we might do so by using the MaxEnt semantics [[3]](#ref-3).

```python
>>> P = pasp.parse("examples/game.lp")
>>> pasp.exact(P, psemantics = "maxent")
```
```
ℙ(wins(b)) = [0.850000, 0.850000]
ℙ(wins(c)) = [0.300000, 0.300000]
```

## Installation and requirements

`pasp` requires Python version 3.10 or newer to work and needs access to
[`clingo`](https://potassco.org/)'s C API. Some Linux distribution packages for `clingo` do not
expose headers or are outdated. Here are some packages we know work with `pasp`.

### Ubuntu PPA

`clingo` and `libclingo-dev`:

```bash
sudo add-apt-repository ppa:potassco/stable
sudo apt update
sudo apt-get install clingo libclingo-dev
```

### Mac OS X

Homebrew `clingo`:

```bash
brew install clingo
```

You may need to append Homebrew to your paths if you have not done so yet:

```bash
# Assuming your brew dir is "${HOME}/.brew"
export LIBRARY_PATH="${LIBRARY_PATH}:$(brew --prefix)/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(brew --prefix)/lib"
export C_INCLUDE_PATH="${C_INCLUDE_PATH}:$(brew --prefix)/include"
```

### Installation

`pasp` is available from the PyPi repository as `pasp-plp`. For Arch Linux users, see the
`python-pasp` package below.

```bash
pip install pasp-plp
```

Import the package normally

```python
>>> import pasp
```

to have access to the exported symbols of the package.

### Arch Linux AUR

The [`python-pasp`](https://aur.archlinux.org/packages/python-pasp) package is available for Arch
Linux users (replace `yay` with your AUR helper or manually install with `makepkg`):

```bash
yay -S python-pasp
```

This installs `pasp` and all its dependencies.

### Manual installation

Alternatively, you may locally build `pasp`. To do so, clone this repository to a directory of your
choice, say `pasp/`. The package is written in Python, with the more critical parts written as C
extensions. The only dependency from the C-side of `pasp` is [clingo](https://potassco.org/), while
the only dependencies from the Python side are the [clingo](https://potassco.org/) Python API and
[lark](https://github.com/lark-parser/lark). Change your working directory to `pasp/` and then
compile and install the C parts with the following command:

```bash
python setup.py build
python setup.py install
```

## Command-line usage

`pasp` is also available as a command:

```bash
% Prints the help message.
pasp --help

% Runs the prisoners example with credal inference and stable semantics.
pasp examples/prisoners.lp

ℙ(e1 | u) = [0.290426, 0.379192]
ℙ(e1 | not b, u) = [0.450125, 0.549875]
ℙ(g | e1, u) = [0.000000, 1.000000]
ℙ(d) = [0.000000, 1.000000]
ℙ(e1 | g, u) = [0.000000, 0.549875]
ℙ(e1 | ga, u) = [0.279971, 0.390743]

% Runs the 3-coloring example with credal inference and L-stable semantics.
pasp --sem=lstable examples/3coloring.lp

ℙ(c(1,r)) = [0.000000, 1.000000]
ℙ(e(1,2) | undef f) = [0.772727, 0.772727]
ℙ(undef f) = [0.064453, 0.064453]

% Runs the insomnia example with Max-Entropy inference and stable semantics.
pasp --psem=maxent examples/insomnia.lp

ℙ(insomnia) = [0.300000, 0.300000]
ℙ(work) = [0.650000, 0.650000]
ℙ(sleep) = [0.350000, 0.350000]
ℙ(not sleep) = [0.650000, 0.650000]
ℙ(not work) = [0.350000, 0.350000]
```

## References

<div id="ref-1">[1] - The Joy of Probabilistic Answer Set Programming: Semantics, Complexity, Expressivity,
Inference. Fabio Gagliardi Cozman and Denis Deratani Mauá. In International Journal of Approximate
Reasoning 125. 2020.</div>
<br>

<div id="ref-2">[2] - On the Semantics and Complexity of Probabilistic Logic Programs. Fabio Gagliardi Cozman and
Denis Deratani Mauá. In Journal of Artificial Intelligence Research 60. 2017.</div>
<br>

<div id="ref-3">[3] - Probabilistic Reasoning with Answer Sets. Chitta Baral, Michael Gelfond and
Nelson Rushton. In International Conference on Logic Programming and Nonmonotonic Reasoning. 2004.</div>
