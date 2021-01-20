from language_model import CorpusReader, LanguageModel, WikipediaCorpusReader
from permutation import Permutation
from simulated_annealing import SimulatedAnnealing


def main():
    model = create_model()
    message = read_message()
    message_chars = "".join(set(message))
    base_hypothesis = Permutation(sigma=model.corpus.sigma(), message_sigma=message_chars)

    run_multiple_simulation(base_hypothesis, message, model, temp_values=[1000], cool_rate_values=[0.9995], threshold_values=[10e-6])


def run_multiple_simulation(base_hypothesis, message, model, temp_values=(10, 100, 1000),
                            cool_rate_values=(0.95, 0.995, 0.9995),
                            threshold_values=(10 ** -1, 10 ** -3, 10 ** -5)):
    for temp in temp_values:
        for cool_rate in cool_rate_values:
            for threshold in threshold_values:
                hypothesis = run_simulation(cool_rate, base_hypothesis, message, model, temp, threshold)
                print("temp=",temp, "threshold=", threshold,"coolrate=", cool_rate,
                      "\n", repr(hypothesis.translate(message)))


def run_simulation(cool_rate, hypothesis, message, model, initial_temp, threshold):
    simulated_annealing = SimulatedAnnealing(initial_temp, threshold, cool_rate)
    hypothesis = simulated_annealing.run(hypothesis, message, model)
    return hypothesis


def read_message():
    return "zv jke nvao ak vahrho vnpurxnho ak ,htyrui vjkunu"
    return open("problemset_07_encrypted_input.txt").read()


def create_model():
    # corpus = CorpusReader("http://www.gutenberg.org/files/76/76-0.txt")
    corpus = WikipediaCorpusReader()
    model = LanguageModel(corpus)
    return model


if __name__ == '__main__':
    main()
