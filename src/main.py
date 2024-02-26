#!/usr/bin/env python3

from image_manage import from_emotions_to_image
import numpy as np
from hmmlearn import hmm
from enum import Enum
import functools

class Humeur(Enum):
    JOYEUX = 0
    TRISTE = 1

# np.random.seed(42)

def compute_state_probability(pred, probs):
    return functools.reduce(lambda x, y: x*y,(map(lambda x, y: y[x.value], pred, probs)))

def read_humour(i: int) -> Humeur:
    return Humeur.TRISTE if i == 1 else Humeur.JOYEUX

teacher_model = hmm.CategoricalHMM(n_components=2, n_iter=3)

# Setting up the starting probabilities
teacher_model.startprob_ = np.array([0.6, 0.4])

# Setting up the transition matrix
transition = np.array([[0.6, 0.4], [0.3, 0.7]])
teacher_model.transmat_ = transition


teacher_model.emissionprob_ = np.array([[0.7,  0.2,  0.1],
                                       [0.05, 0.15, 0.8]])

observations, states = teacher_model.sample(n_samples=3)

print("Observed: ", *observations)
print(teacher_model.predict_proba(observations))

from_emotions_to_image(observations)

possible_states = [[a, b, c] for a in Humeur
                             for b in Humeur for c in Humeur]


matrix_with_probas = [(poss, compute_state_probability(poss,
                            teacher_model.predict_proba(observations)))
                      for poss in possible_states]


matrix_with_probas.sort(key=lambda x: x[1])
matrix_with_probas.reverse()


for a, b in matrix_with_probas:
    print(*a, "has probability", b)


print("\nComputer's best prediction: " ,*map(read_humour, teacher_model.predict(observations)))
print("Real Answer: ", *map(read_humour, states))
