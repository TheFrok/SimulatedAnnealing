import urllib.request
import string
import re
from collections import Counter
import wikipedia as wiki
from wikipedia.exceptions import WikipediaException


def get_wikidata(start_page, max_page_count=100, lang="en"):
    pages = set()
    batch = [start_page]
    data = ""
    wiki.set_lang(lang)
    while len(pages) < max_page_count and len(batch) > 0:
        title = batch.pop()
        try:
            page = wiki.page(title)
            new = set(page.links)
            data += page.content + " "
        except WikipediaException:
            continue
        pages.add(title)
        batch = list(new.difference(pages)) + batch
    return data, pages


class CorpusReader:
    # All accepted letters in the corpus
    CHARACTERS = string.ascii_lowercase
    NON_CHAR = " ,.:\n#()!?'\""

    def __init__(self, url):
        self.url = url
        self.data = ""
        self.read_filtered_url()

    def sigma(self):
        return self.CHARACTERS + self.NON_CHAR

    def _get_replace_pattern(self):
        sigma = self.CHARACTERS + self.NON_CHAR
        # compiled regex to clean the corpus
        return re.compile(f"[^{sigma}]")

    def read_url(self):
        return urllib.request.urlopen(self.url).read()

    def filter_data(self, data: str):
        data = data.lower()
        return self._get_replace_pattern().sub("", data)

    def read_filtered_url(self):
        data = self.read_url().decode("utf8")
        self.data = self.filter_data(data)


class WikipediaCorpusReader(CorpusReader):
    CHARACTERS = "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
    DEFAULT_WIKIPEDIA_ARTICLE = "תאטרון החלומות"

    def __init__(self, article=DEFAULT_WIKIPEDIA_ARTICLE, lang="he"):
        self.data, self.pages = get_wikidata(article, lang=lang)
        self.data = self.filter_data(self.data)


class LanguageModel:
    def __init__(self, corpus: CorpusReader):
        self.corpus = corpus
        # containers for the unigrams and bigrams counts
        self.unigram_counts: Counter = Counter()
        self.bigram_counts: Counter = Counter()
        # calculate counts
        self.get_counts()
        # containers for the probabilities
        self.unigram_probs: dict = {}
        self.bigram_probs: dict = {}
        # calculate probabilities from the counts
        self.calc_probabilities()

    def get_counts(self):
        self.unigram_counts.update(self.corpus.data)
        # generate bigrams from a text - https://stackoverflow.com/a/12488794/6136361
        bigrams = zip(self.corpus.data, self.corpus.data[1:])
        self.bigram_counts.update(bigrams)

    def get_probabilty(self, letter, assuming=None) -> float:
        """
        return the probability of P(w1|w2) when assuming is w2.
        If its null then calculting P(w1). Both probabilities are based on the counts after laplace
        :param letter: the letter
        :param assuming: the previous letter / None
        :return: the probability of `letter` given that `assuming` is the previous letter
        """
        if not assuming:
            return self.unigram_probs[letter]
        return self.bigram_probs[(letter, assuming)]

    def calc_probabilities(self) -> None:
        sigma = self.corpus.sigma()
        unigram_size = len(self.unigram_counts) + len(sigma)
        for char1 in sigma:
            # calculate probability of `char1` as the first letter in the text
            c_count = self.unigram_counts[char1]
            self.unigram_probs[char1] = LanguageModel.laplace_prob(c_count, unigram_size)
            # calculate the probability of `char2` given `char1` is the previous letter
            for char2 in sigma:
                bigram = (char2, char1)
                bi_count = self.bigram_counts[bigram]
                # use the max_page_count of `char1` as the total size of all pairs starting with `char1`
                self.bigram_probs[bigram] = LanguageModel.laplace_prob(bi_count, c_count)

    @staticmethod
    def laplace_prob(count, corp_size) -> float:
        return (count + 1) / (corp_size + 1)


if __name__ == '__main__':
    # corp = CorpusReader("http://www.gutenberg.org/files/76/76-0.txt")
    corp = WikipediaCorpusReader()
    model = LanguageModel(corp)
    print(model.bigram_counts.most_common(10))
    print(model.get_probabilty('ת', ' '))
    print(model.get_probabilty(',', 'ב'))
