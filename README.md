# Mini DCM Toy Model

This project implements a minimal 3-node Dynamic Causal Model to demonstrate
Bayesian model comparison over causal connectivity structures.

## Usage

python main_demo.py

## Model

The system is linear and Gaussian:

x(t+1) = A x(t) + C u(t) + w(t)  
y(t)   = x(t) + v(t)

The ground-truth connectivity is a simple chain:

1 → 2 → 3  
with self-connections on all nodes and input (stimulus) applied only to node 1.

## Competing Models

A model is defined by a boolean mask on A indicating which connections are free
parameters and which are fixed to zero. Three masks are compared:

- **M1_chain_true** — correct chain 1→2→3  
- **M2_missing_2to3** — same chain but 2→3 removed  
- **M3_wrong_3to1** — incorrect feedback 3→1, no 2→3

Each mask determines a different regression structure and number of parameters.

## Demo

`main_demo.py` runs the full pipeline:

- simulate data from the true model  
- fit all candidate models  
- compute evidence for each  
- print posterior probabilities