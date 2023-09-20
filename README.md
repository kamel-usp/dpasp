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

## Acknowledgment

This software is being developed by the KAMeL group of the University of São Paulo and the Center for Artificial Intelligence.
If you use this software, please acknowledge by citing the paper below:

  https://arxiv.org/abs/2308.02944
