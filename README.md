# bandit-algorithms
A small collection of Bandit Algorithms (ETC, E-Greedy, Elimination, UCB, Exp3, LinearUCB, and Thompson Sampling)

Python implementation of selected multi-arm bandit algorithms that are fundamental to recommendation systems, online advertising, clinical trials, online resource allocation and online search.

Vis-a-vis Bandits used in Reinforcement Learning settings, three key assumptions apply:
- Reward observed only conrrespond to the action taken (feedback).
- Action does not alter the environment.
- Taking an action does not restrict future actions.

Furthermore, these algorithms differ in their requirement of time horizon and reward gaps, and in their scaling of regret bounds.



| Algorithm | Environment | Requirement | Regret Scaling |
| --- | --- | --- | --- |
| Explore-then-Commit | Unstructured | ![\sum_{\forall i}{x_i^{2}}](https://latex.codecogs.com/svg.image?\Delta&space;) | --- |
| Epsilon-Greedy | --- | --- | --- |
| Epsilon-Greedy | --- | --- | --- |
| Epsilon-Greedy | --- | --- | --- |
| Epsilon-Greedy | --- | --- | --- |
| Epsilon-Greedy | --- | --- | --- |
| Epsilon-Greedy | --- | --- | --- |
