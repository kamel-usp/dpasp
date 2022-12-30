# An overview of pasp {#mainpage}

[![Tests](https://github.com/RenatoGeh/pasp/actions/workflows/tests.yml/badge.svg)](https://github.com/RenatoGeh/pasp/actions/workflows/tests.yml)
[![Docs](https://github.com/RenatoGeh/pasp/actions/workflows/docs.yml/badge.svg)](https://github.com/RenatoGeh/pasp/actions/workflows/docs.yml)
[![](https://img.shields.io/badge/docs-master-blue.svg)](https://renatogeh.github.io/pasp)
[![GitHub](https://img.shields.io/github/license/RenatoGeh/pasp?color=blue&label=License)](https://github.com/RenatoGeh/pasp/blob/master/LICENSE)

This is the **developer** documentation for `pasp`, which includes both the Python and C APIs, and
references code which is invisible to the user. See the [**user** documentation](../index.html) if
you are uninterested in reading a ten-page essay on how to properly deallocate memory.

This page offers a brief birds-eye overview of `pasp`, presenting the project structure, details on
compilation, and use of notation, nomenclature and conventions in the code. This page should be
read *before* the subsequent documentation on the specificities of each module.

---

## Project structure

`pasp` is divided into a front-end (the less interesting exposed Python API), and a back-end (the
more interesting C API). To bridge the gap between the two, we use the [Python C Extension
API](https://docs.python.org/3/c-api/), which you should be acquainted with before reading these
docs -- here's some [light reading](https://docs.python.org/3/extending/index.html) before going to
bed.

The Python API mainly serves two purposes: (1) provide a friendly shell to the user, and (2) do the
grammar parsing, which we shall talk about [later down the line](grammar.md). There are currently
`6` modules in `pasp`:

1. `program` [is in charge of program definitions](@ref program_md);
2. `grammar` [takes care of parsing and grammar functions](@ref grammar_md);
3. `ground`  [does pre-grounding for probabilistic rules with variables](@ref ground_md);
4. `exact`   [provides exact inference functions](@ref exact_md);
5. `sample`  [gives access to sampling functions](@ref sample_md);
6. `learn`   [implements parameter learning](@ref learn_md).

The last four are implemented in C, and provide access to functions via the Python-C API, while
the first two are written purely in Python. Let us first briefly describe these two.

Module `program` mostly serves as a user interface between the hidden C API and what is visible
from the Python end. Internally, all the structures defined in this module are converted into an
equivalent C version. How these structures are initially constructed, however, comes from
[parse](@ref pasp.grammar.parse), which builds a [Program](@ref pasp.program.Program) from text.

Module `grammar` [parse](@ref pasp.grammar.parse)s either a file or a string in accordance with a
formal grammar definition as a [.lark](https://lark-parser.readthedocs.io/en/latest/) extension
file (see [grammar.lark](https://github.com/RenatoGeh/pasp/blob/master/pasp/grammar.lark)),  and
assembles a program given their contents. Because programs are usually small, running parsing in
Python is more than enough in our use case.

Once a program has been constructed, the user might want to process any queries. For now, `pasp`
only admits exact inference by exhaustive model counting. Exact inference is called from the
`exact` module, written in C for performance purposes. However, running inference requires a
*pre-grounded* program, i.e. a program whose probabilistic atoms have all been grounded.

Module `ground` grounds all probabilistic rules with variables in the program, in a process we call
*pre-grounding*, done partially in [clingo](https://potassco.org/clingo/). A pre-grounded program
is then injected into the program structure for use in `exact`.

Functions for randomly sampling atoms from programs reside in the `sample` module.

Parameter learning is done in the `learn` module.
