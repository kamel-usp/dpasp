# dPASP: A Flexible Framework For Neuro-Probabilistic Answer Set Programming

[![Tests](https://github.com/kamel-usp/dpasp/actions/workflows/tests.yml/badge.svg)](https://github.com/kamel-usp/dpasp/actions/workflows/tests.yml)
[![Docs](https://github.com/kamel-usp/dpasp/actions/workflows/docs.yml/badge.svg)](https://github.com/kamel-usp/dpasp/actions/workflows/docs.yml)
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://kamel-usp.github.io/dpasp)
[![GitHub](https://img.shields.io/github/license/kamel-usp/dpasp?color=blue&label=License)](https://github.com/kamel-usp/dpasp/blob/master/LICENSE)

The dPASP framework presents a powerful high-level language for describing
probabilistic tasks in an intuitive and declarative manner. Just like in traditional
probabilistic logic programming (PLP), programs in dPASP are written in terms of
probabilistic facts or rules, allowing for uncertainty to play a role in the knowledge
description of the problem. Notably, our framework further extends PLP by leveraging the
expressiveness of neural networks for describing probabilities in possibly hybrid domains.
Further, by natively embedding neural expressions within the language, dPASP offers
end-to-end training of sophisticated models and loss functions while requiring minimal user
knowledge of deep learning system's inner workings.

## Getting Started 

dPASP is mostly written in C with a front user layer in Python. Although a Python API
is exposed for manipulating programs and their output, dPASP can be used as a
standalone language and interpreter. 

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

Assuming you have dPASP installed and configured, open a Python Shell and load the library by:

```
>>> import pasp
>>> program_str = '''
#const n = 5.
v(1..n).
0.5::e(X, Y) :- v(X), v(Y), X < Y.
e(X, Y) :- e(Y, X).
c(X, r) :- not c(X, g), not c(X, b), v(X).
c(X, g) :- not c(X, r), not c(X, b), v(X).
c(X, b) :- not c(X, r), not c(X, g), v(X).
f :- not f, e(X, Y), c(X, Z), c(Y, Z).
#semantics maxent.
'''
>>> P = pasp.parse(program_str, from_str=True)
```
Note the directive `#semantics maxent` selecting the respective probabilistic semantics (other options are `partial`, `lstable`, `credal`).

You can check that the program was correctly parsed by verifying the object `P`

```
>>> P()
```

To run exact (enumerative) inference, do:
```
>>> pasp.exact(P)
```

You can also run inference with a specified semantics:
To run exact (enumerative) inference, do:
```
>>> pasp.exact(P, psemantics="credal")
```

## Acknowledgments

This software is being developed by the KAMeL group of the University of São Paulo and the Center for Artificial Intelligence.
If you use this software, please acknowledge by citing the paper below:

  https://arxiv.org/abs/2308.02944
