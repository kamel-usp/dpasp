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
- Partial semantics;
- L-Stable semantics.

### Probabilistic semantics

- Credal semantics;
- MaxEnt semantics.

There are two uses of the systems: learning and querying. 
Currently, learning allows only for MaxEnt-Stable semantics.

Learning and querying can either be made by (highly inneficient) enumerative algorithms and (more efficient) approximate inference.
Enumerative algorithms are available to all possible (logic and probabilistic) semantics, while currently only MaxEnt-Stable semantics implements all approximate algorithms.
Developing more efficient and accurate approximate algorithms is a current active line of research.

## Example

Here's a simple example of dPASP for inference in probabilistic logic programs (no neural networks and no learning). 

Assuming you have dPASP installed and configured (see the [tutorial](http://kamel.ime.usp.br/pages/learn_dpasp) if not), open a Python Shell and load the Python library by:

```python
import pasp
```

We can specify a probabilistic logic program string using dPASP DSL.
Here a program enconding graph 3-coloring property (lines starting with `%` are comments):

``` python
program_str = '''
% DOMAIN
#const n = 5.
vertex(1..n).
% Specifies a random graph
0.5::edge(X, Y) :- vertex(X), vertex(Y), X < Y.
edge(X, Y) :- e(Y, X).
% Disjunctive specify specify candidate solutions 
color(X, red); color(X,blue); color(X,green) :- vertex(X).
% Constraints discard invalid candidate solutions
:- edge(X, Y), color(X, Z), color(Y, Z).
% We use directives to select a semantics (other options are `partial`, `lstable`, `credal`)
#semantics maxent.
% Directive also specify the query, in this case the probability that node 1 is colored red
#query color(1,red).
'''
```

To parse the program into dPASP internal's data struct we use
```python
P = pasp.parse(program_str, from_str=True)
```

You can check that the program was correctly parsed by inspecting the object `P`

```python
>>> P()
```

To run exact (enumerative) inference, do:
```
pasp.exact(P)
```

You can also run inference with a specified semantics:
To run exact (enumerative) inference, do:
```
pasp.exact(P, psemantics="credal", semantics="stable")
```

## Acknowledgments

This software is being developed by the [KAMeL group](https://kamel.ime.usp.br) and the [Center for Artificial Intelligence](https://c4ai.inova.usp.br/) of the University of São Paulo.
If you use this software, please acknowledge by citing the paper below:

  https://arxiv.org/abs/2308.02944
