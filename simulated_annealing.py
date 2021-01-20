import math
import random

from language_model import LanguageModel
from permutation import Permutation


class SimulatedAnnealing:
    def __init__(self, init_temp: float, threshold: float, cooling_rate: float):
        self.init_temp = init_temp
        self.threshold = threshold
        self.cooling_rate = cooling_rate

    def run(self, hypothesis: Permutation, message: str, model: LanguageModel):
        temp = self.init_temp
        while temp > self.threshold:
            neighbor = hypothesis.get_neighbor()
            energy = hypothesis.get_energy(message, model)
            delta = neighbor.get_energy(message, model) - energy

            # the probability of switching to the neighbor
            p = self.neighbor_probability(delta, temp)
            if random.random() < p:
                hypothesis = neighbor
            # cool the system
            temp = self.cooling_rate * temp
        return hypothesis

    @staticmethod
    def neighbor_probability(delta, temp):
        if delta < 0:
            p = 1
        else:
            p = math.exp(- delta / temp)
        return p
