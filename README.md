# natsune

Higher Order Concurrency in Python

## What is this?

A an experimental project to compile python to interaction nets, hosted in the python interpreter, in order to gain numerous interest concurrency properties.

For instance, time travel.  Notice how smallest is set at the end of the function, yet propagates backwards through the loop.

```python
@inet
def shift_list_by_smallest(l: list[int]) -> list[int]:
    if len(l) == 0:
        return []

    smallest: Inverse[int]
    smallest_acc: int = l[0]

    result: Ref[list[int]] = []

    for v in l:
        if v < smallest_acc:
            smallest_acc = v
        print(smallest)
        result.append(v - smallest)

    smallest = smallest_acc

    return result


assert shift_list_by_smallest([4, 9, 1, 10]) == [3, 8, 0, 9]
```


## What about other interaction net languages?

They're awesome!  And most of them are focused on first order concurrency -- writing decidable software as fast as possible.

That's not my vision for natsune.  First off, because those other projects exist, if you know how to solve your problem at the first order, you should just contribute to and use those projects.

Natsune, however, runs in the python interpreter.  In the near future, it will support GIL-less concurrency.  It's goal is to be usable alongside the entire ecosystem of powerful serialized first order python tools, but gain the ability express higher order concurrency without large code rewrites.  You can call into and out of natsune from standard python code much like async, generators, or stackless python approaches.

Unlike those approaches, however, the coloring problem is subtly shifted away from explicit scheduling to the type system: describing which values are `Par` (independent variables), which are `References` (linearized contingents), which are `Inverse` (computational Futures), and which are plain old serialized values (which can either `Copy` or `Share`).

In natsune, scheduling is non generalizable.  The focus is on the logic and the contingencies.  The ambiguity is in the time.

## Why?

<a href="https://journals.sagepub.com/doi/10.1177/10597123241269740">Because of this very modest paper</a>.

Listen,  Machine Learning is valuable because it is attacking the important part of math: the latent space.  But it's definitely NOT THE ONLY WAY to do that.

My hypothesis.  Bundles of arbitrary algotypes can also explore latent spaces and do higher order reasoning in very simple, computationally cheap ways that traditional first order computation in practice won't (**CAN** is not the interesting question, **HOW** is).

My desire is to fully explore this idea: can I reconstruct calculus and a vast number of interesting scientific computing in a discrete, concurrent way.  No statistics.  No training.  Just the surprising consequence of subtle and simple descriptions.
