# OSD: Optimization(-based) Signal Decomposition
_Modeling language for finding signal decompositions_

This software provides a modeling language for describing and solving signal decomposition problems. This framework is described in detail in an acompanying [monograph](https://web.stanford.edu/~boyd/papers/sig_decomp_mprox.html). Examples are available in the [notebooks](notebooks/) directory.

## Brief overview

We formulate the problem of decomposing a signal into components as an optimization problem, where components are described by their loss functions. Once the component class loss functions are chosen, we minimize the total loss subject to replicating the given signal with the components. Our software provides a robust algorithm for carying out this decomposition, which is guaranteed to find the globally optimal descomposition when the loss functions are all convex, and is a good heuristic when they are not.

## Vector times series signals with missing entries

We consider a vector time series or signal, `y`, which possibly has missing entries. We represent the signal compactly as a `T x p` matrix, with `T` time steps and `p` entries at each time. Some of these entries may be missing—filled with `NaN` or some other missing value indicator. We will be concerned with finding decompositions that exactly match the data at the known entries.

## Signal decomposition

We model the given signal `y` as a sum (or decomposition) of `K` component, `x^1,...,x^K`. Each component `x^k` is also a `T x p` matrix, but they do not have any missing values. Indeed, we can use the values of `x^1,...,x^K` as estimates of the missing values in the original signal. This is useful for data imputation as well as model validation.

## Component classes

The `K` components are characterized by cost functions that encode the loss of or impolausibility that a component takes on a certaint value. (In some cases, we can interpret the classes statistically, with the cost function corresponding to the negative log-likelihood of some probability density function, but this is not necessary.) Our solution method is based on evaluating the _masked proximal operators_ of the class cost functions. These operators have been defined for many useful classes in the [classes](osd/classes/) module.

# Installation

We do not yet have a package released for this code, so for now, please clone the repository and set up a virtual environment with the packages listed in the [requirements file](requirements.txt).
