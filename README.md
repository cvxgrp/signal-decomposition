# optimal-signal-decomposition
_Modeling language for finding signal decompositions_

This software provides a modeling language for describing and solving optimal signal demixing problems. The software solves problems of the form:

```
y = x_0 + x_1 + ... + x_K
```

where y is some signal. Each `x_k` is assumed to be defined by a "signal class" which is a cost function that can be interpreted as the negative log-likelihood of observing the constituent signal. When all signal class functions are convex, the demixing problem is a convex optimization problem and can be solved exactly. All signal classes implemented so far are convex. Eventually, this project will cover non-convex cases as well.

The goal of the modeling language is to abstract out the math from solving this type of problem. 

See [notebooks](notebooks/) for examples.

A manuscript on this methodology is nearing completion.

Some additional background can be found here:

- [https://bmeyers.github.io/source_separation_convex_opt/](https://bmeyers.github.io/source_separation_convex_opt/)
- [https://bmeyers.github.io/QuantileRegression/](https://bmeyers.github.io/QuantileRegression/)
