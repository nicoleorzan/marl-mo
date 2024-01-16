# Multi-Agent Multi-Objective Reinforcement Learning for the EPGG


Create a virtual environment as follows:

```bash
python3 -m venv env

```
and activate it:

```bash
source env/bin/activate

```
Don't forget to update pip:


```bash
pip install --upgrade pip

```

Install the setup file as follows (the -e options allows to change the code while using it):

```bash
pip install -e .
```

Install the required packages:

```bash
pip install -r requirements.txt
```

To call the code:
```
python3 caller_optuna_anastassacos.py --n_agents 2 --uncertainties 0. 0. --communicating_agents 0 0 --listening_agents 0 0 --gmm_ 0 --algorithm reinforce --proportion_dummy_agents 0.5 --opponent_selection 0 --binary_reputation 0 --b_value 5
```


