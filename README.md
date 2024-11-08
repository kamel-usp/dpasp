# dPASP: A Flexible Framework For Neuro-Probabilistic Answer Set Programming

[![Tests](https://github.com/kamel-usp/dpasp/actions/workflows/tests.yml/badge.svg)](https://github.com/kamel-usp/dpasp/actions/workflows/tests.yml)
[![Docs](https://github.com/kamel-usp/dpasp/actions/workflows/docs.yml/badge.svg)](https://github.com/kamel-usp/dpasp/actions/workflows/docs.yml)
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://kamel-usp.github.io/dpasp)
[![GitHub](https://img.shields.io/github/license/kamel-usp/dpasp?color=blue&label=License)](https://github.com/kamel-usp/dpasp/blob/master/LICENSE)


The dPASP framework provides a high-level declarative language for describing
sophisticated probabilistic reasoning tasks that can combine perception and logical reasoning.
dPASP programs are made of probabilistic choices and logic rules (think of if-then rules), allowing
easy specification of uncertain and logic knowledge. Notably, the probabilities in the program
can arise as outputs of neural network classifiers, and both the statistical model and logic program can be jointly optimized
 by making use of efficient gradient-based learning libraries (e.g. PyTorch).

## Getting Started

dPASP has both a domain-specific language (DSL) and command-line interpreter (parser) for that language, which can be used as
a standalone tool. Alternatively, dPASP can be accessed as Python library or more directly through its C backend.

The easiest way to get started is by reading the tutorial [Learning dPASP Through Examples](http://kamel.ime.usp.br/pages/learn_dpasp).

## Features

dPASP allows for several semantics by combining logic programming semantics and probabilistic semantics:

### Logic semantics

- Stable semantics;
- Partial Stable semantics;
- L-Stable semantics;
- smProbLog semantics.

### Probabilistic semantics

- Credal semantics;
- MaxEnt semantics.

There are two uses of the systems: learning and querying.
Currently, learning is only available for the MaxEnt-Stable semantics.

Learning and querying can either be made by (highly inneficient) enumerative algorithms and (more efficient) approximate inference.
Enumerative algorithms are available to all possible (logic and probabilistic) semantics, while currently only MaxEnt-Stable semantics implements all approximate algorithms.
Developing more efficient and accurate approximate algorithms is a current active line of research.

## Example

Here's a simple example of dPASP for inference in probabilistic logic programs (no neural networks and no learning).

Assuming you have dPASP installed and configured (see the [tutorial](http://kamel.ime.usp.br/pages/learn_dpasp) if not), open a Python Shell and load the Python library by:

```bash
pasp examples/earthquake.plp
```

One may need to manipulate output probabilities, in which case the Python API can be used instead.

```python
import pasp
```

To parse the program into dPASP internal's data struct we use

```python
P = pasp.parse("examples/earthquake.plp")
```

We can also specify a probabilistic logic program string using the dPASP DSL.
Here we have a program enconding graph 3-coloring (lines starting with `%` are comments):

``` python
program_str = '''
% Build a random graph with n vertices.
#const n = 5.
v(1..n).
% The choice of p reflects the sparsity/density of the random graph.
% A small p produces sparser graphs, while a large p prefers denser graphs.
0.5::e(X, Y) :- v(X), v(Y), X < Y.
e(X, Y) :- e(Y, X).
% A color (here the predicate c/2) defines a coloring of a vertex.
% The next three lines define the uniqueness of a vertex's color.
c(X, r) :- not c(X, g), not c(X, b), v(X).
c(X, g) :- not c(X, r), not c(X, b), v(X).
c(X, b) :- not c(X, r), not c(X, g), v(X).
% Produce a contradiction if two neighbors have the same color.
f :- not f, e(X, Y), c(X, Z), c(Y, Z).

#semantics lstable.

% Query the probability of vertex a being red.
#query(c(1, r)).
% Query the probability of the edge e(1, 2) existing given that the graph is not 3-colorable.
#query(e(1, 2) | undef f).
% Query the probability of the graph not being 3-colorable.
#query(undef f).'''
```

We may then pass `program_str` to dPASP as a string by using the `from_str=True` keyword.

```python
P = pasp.parse(program_str, from_str=True)
```

You can check that the program was correctly parsed by inspecting the object `P`

```python
>>> P
<Logic Program:
#const n = 5.
v(1..n).
e(X, Y) :- e(Y, X).
_e(X, Y) :- _e(Y, X).
c(X, r) :- not _c(X, g), not _c(X, b), v(X).
_c(X, r) :- not c(X, g), not c(X, b), _v(X).
c(X, g) :- not _c(X, r), not _c(X, b), v(X).
_c(X, g) :- not c(X, r), not c(X, b), _v(X).
c(X, b) :- not _c(X, r), not _c(X, g), v(X).
_c(X, b) :- not c(X, r), not c(X, g), _v(X).
f :- not _f, e(X, Y), c(X, Z), c(Y, Z).
_f :- not f, _e(X, Y), _c(X, Z), _c(Y, Z).
_c(X, g) :- c(X, g).
_e(X, Y) :- e(X, Y).
_c(X, b) :- c(X, b).
_f :- f.
_c(X, r) :- c(X, r).
_v(1..n) :- v(1..n).,
Probabilistic Rules:
[0.5::e(X, Y) :- v(X), v(Y), X < Y],
Queries:
[ℙ(c(1,r)), ℙ(e(1,2) | undef f), ℙ(undef f)]>
```

To run the program:
```python
>>> P()
ℙ(c(1,r)) = [0.000000, 1.000000]
ℙ(e(1,2) | undef f) = [0.772727, 0.772727]
ℙ(undef f) = [0.064453, 0.064453]
---
Querying:                                                               0h00m01s
array([[0.        , 1.        ],
       [0.77272727, 0.77272727],
       [0.06445312, 0.06445312]])
```

## Acknowledgments

This software is being developed by the [KAMeL group](https://kamel.ime.usp.br) and the [Center for Artificial Intelligence](https://c4ai.inova.usp.br/) of the University of São Paulo.
If you use this software, please acknowledge by citing the paper below:

  https://proceedings.kr.org/2024/69/
