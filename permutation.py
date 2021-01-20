import math
import random
import string

from language_model import CorpusReader, LanguageModel


class Permutation:
    def __init__(self, permutation=None, sigma=string.ascii_lowercase, message_sigma=None):
        self.sigma = sigma
        if message_sigma is None:
            message_sigma = sigma
        self.source_sigma = message_sigma
        self.perm = list(permutation) if permutation else list(self.sigma)

    def __repr__(self):
        return f"Permutation({repr(''.join(self.perm))})"

    def get_neighbor(self):
        a, b = random.sample(range(len(self.perm)), 2)
        new_perm = list(self.perm)
        new_perm[a] = self.perm[b]
        new_perm[b] = self.perm[a]
        return Permutation(new_perm, sigma=self.sigma, message_sigma=self.source_sigma)

    def translate(self, input: str):
        # create a trans table for the str.translate method
        cut_perm = ''.join(self.perm)[:len(self.source_sigma)]
        trans_table = str.maketrans(self.source_sigma, cut_perm)
        # translate using python built-in translate
        return input.translate(trans_table)

    def get_energy(self, data: str, model: LanguageModel):
        guess = self.translate(data)
        # initiate energy with the log of probability of the first letter
        energy = math.log(model.get_probabilty(guess[0]), 2)
        # add the log of probability of the bigrams in the texr
        for letter, assuming in zip(guess[:-1], guess[1:]):
            prob = model.get_probabilty(letter, assuming)
            energy += math.log(prob, 2)
        # return -energy to search for minimum instead of maximum
        return -1 * energy


if __name__ == '__main__':
    p = Permutation()
    print(p)
    for i in range(20):
        p = p.get_neighbor()
        print(p)
