# Multi-Objective Multi-Agent Reinforcement Learning in Public Goods Games

**Paper:** *Learning in Multi-Objective Public Goods Games with Non-Linear Utilities* (ECAI 2024)  
Authors: Nicole Orzan, Erman Acar, Davide Grossi, Roxana Rădulescu  
[📄 OpenReview](https://openreview.net/forum?id=1GXIiEo9wj)

---

### TL;DR (paper, in a few lines)

- We introduce MO-EPGG, a **multi-objective** version of the Extended Public Goods Game that separates **individual vs collective payoffs**.
- Each agent uses a non-linear utility (risk preference) over the payoff vector, and we study how this interacts with incentive-alignment uncertainty. 
- With MORL we show regimes where preferences + uncertainty promote or suppress cooperation even in mixed-motive settings. 

### Example Install

```
python3 -m venv env
source env/bin/activate
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

To launch a trial run, in `src/experiments/` give the comman:

```
python3 caller.py
```
This defaults to running a 2-agents 2-objectives MO-EPGG, with MO-deep-Q Learning for each agent and linear scalarization function. Other default parameters can be found in the file.

For a custom run, you can give:

```
python3 caller.py --n_agents 4 
```

### Citing

If you use this repository, please cite:

```
@inproceedings{orzan2024moepgg,
  title     = {Learning in Multi-Objective Public Goods Games with Non-Linear Utilities},
  author    = {Orzan, Nicole and Acar, Erman and Grossi, Davide and Mannion, Patrick and R{\u{a}}dulescu, Roxana},
  booktitle = {ECAI 2024 - 27th European Conference on Artificial Intelligence, Including 13th Conference on Prestigious Applications of Intelligent Systems (PAIS 2024), Proceedings},
  editor    = {Endriss, Ulle and Melo, Francisco S. and Bach, Kirsten and Bugar{\'\i}n-Diz, Ana and Alonso-Moral, Jos{\'e} M. and Barro, Sen{\'e}n and Heintz, Fredrik},
  series    = {Frontiers in Artificial Intelligence and Applications},
  volume    = {392},
  pages     = {2749--2756},
  publisher = {IOS Press},
  year      = {2024},
  doi       = {10.3233/FAIA240809}
}

```
