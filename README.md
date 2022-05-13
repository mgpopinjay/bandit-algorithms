# bandit-algorithms
A small collection of Bandit Algorithms (ETC, E-Greedy, Elimination, UCB, Exp3, LinearUCB, and Thompson Sampling)

Python implementation of selected multi-arm bandit algorithms that are fundamental to recommendation systems, online advertising, clinical trials, online resource allocation and online search.

Vis-a-vis Bandits used in Reinforcement Learning settings, three key assumptions apply:
- Reward observed only conrrespond to the action taken (feedback).
- Action does not alter the environment.
- Taking an action does not restrict future actions.

Furthermore, these algorithms differ in their requirement of the time horizon ![n](https://latex.codecogs.com/svg.image?n)  and suboptimality gap ![\Delta](https://latex.codecogs.com/svg.image?\Delta_{i}&space;), and in their scaling of regret bounds.



| Algorithm | Environment | Requirement | Regret Scaling |
| --- | --- | --- | --- |
| Explore-then-Commit (ETC) | Unstructured / Stochastic | ![\Delta](https://latex.codecogs.com/svg.image?\Delta_{i}&space;), ![n](https://latex.codecogs.com/svg.image?n) | ![R_{n} = \frac{4}{n}  ln(n) + C](https://latex.codecogs.com/svg.image?R_{n}&space;=&space;\frac{4}{n}&space;&space;ln(n)&space;&plus;&space;C) |
| Epsilon-Greedy (E-Greedy)| Unstructured / Stochastic | ![\Delta](https://latex.codecogs.com/svg.image?\Delta_{i}&space;) | ![R_{n} = \frac{1}{\Delta }  ln(n) + C](https://latex.codecogs.com/svg.image?R_{n}&space;=&space;\frac{1}{\Delta&space;}&space;&space;ln(n)&space;&plus;&space;C) |
| Elimination | Unstructured / Stochastic | ![n](https://latex.codecogs.com/svg.image?n) | ![R_{n} = \frac{C_{1}}{\Delta }  ln(n) + C_{2}](https://latex.codecogs.com/svg.image?R_{n}&space;=&space;\frac{C_{1}}{\Delta&space;}&space;&space;ln(n)&space;&plus;&space;C_{2}) |
| Upper-Confidence Bound (UCB) | Unstructured / Stochastic | ![n](https://latex.codecogs.com/svg.image?n) | --- |
| Exponantiate-Explore-Exploit (Exp3) | Adversarial / Non-stochastic | --- | --- |
| Linear UCB (LinUCB) | Contextual / Linear | --- | --- |
| Thompson Sampling | Structured | --- | --- |
