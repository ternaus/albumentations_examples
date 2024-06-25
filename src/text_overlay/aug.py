import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import RAKE
from nltk.corpus import wordnet
from nltk.tag.stanford import StanfordPOSTagger


class TextAugmenter:
    def __init__(self, stopwords_path: Path, pos_model_path: Path, pos_jar_path: Path, seed: int | None = None) -> None:
        self.stopwords = self.get_stopwords(stopwords_path)
        self.pos_tagger = StanfordPOSTagger(
            model_filename=pos_model_path,
            path_to_jar=pos_jar_path,
            java_options="-mx4000m",
        )
        self.rake = RAKE.Rake(stopwords_path)
        self.rng = np.random.default_rng(seed)  # Initialize the random generator with an optional seed

    @staticmethod
    def get_stopwords(path: Path) -> list[str]:
        with path.open() as f:
            stopwords = f.readlines()
        return [x.strip() for x in stopwords]

    def random_deletion(self, words: list[str], p: float) -> list[str]:
        if len(words) <= 1:
            return words
        new_words = [word for word in words if self.rng.uniform() > p]
        return new_words if new_words else [self.rng.choice(words)]

    def swap_word(self, new_words: list[str]) -> list[str]:
        if len(new_words) < 2:
            return new_words
        random_idx_1 = self.rng.integers(len(new_words))
        counter = 0
        while True:
            random_idx_2 = self.rng.integers(len(new_words))
            if random_idx_1 != random_idx_2:
                new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
                break
            counter += 1
            if counter > 3:
                break
        return new_words

    def random_swap(self, words: list[str], n: int) -> list[str]:
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def get_synonyms(self, word: str) -> list[str]:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
                synonyms.add("".join([char for char in synonym if char.isalpha() or char == " "]))
        synonyms.discard(word)
        return list(synonyms)

    def random_insertion(self, words: list[str], n: int) -> list[str]:
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words: list[str]) -> None:
        non_stopwords = [word for word in new_words if word not in self.stopwords]
        if not non_stopwords:
            return
        random_word = self.rng.choice(non_stopwords)
        synonyms = self.get_synonyms(random_word)
        if synonyms:
            random_synonym = self.rng.choice(synonyms)
            random_idx = self.rng.integers(len(new_words))
            new_words.insert(random_idx, random_synonym)

    def extract_keywords_and_pos(self, prompt: str) -> dict[str, str]:
        pos_dict = {}
        try:
            tagged_prompt = self.pos_tagger.tag(prompt.split())
        except:
            return {}
        for word, pos in tagged_prompt:
            pos_dict[word] = pos
        keywords_dict = {}
        keywords = self.rake.run(prompt)
        for pair in keywords:
            words = pair[0].split()
            for word in words:
                if word in pos_dict:
                    keywords_dict[word] = pos_dict[word]
        return keywords_dict

    def get_new_keyword(self, word: str, pos: str) -> list[str]:
        synonyms: list[str] = []
        try:
            syn_lst = wordnet.synsets(word, pos)
            if not syn_lst:
                syn_lst = wordnet.synsets(word)
        except:
            try:
                syn_lst = wordnet.synsets(word)
            except:
                return synonyms

        # Using list comprehension to collect synonyms
        synonyms = [lemma.name().lower() for syn in syn_lst for lemma in syn.lemmas() if lemma.name().lower() != word]

        return list(dict.fromkeys(synonyms))

    def single_prompt_helper(
        self,
        keywords_lst: list[str],
        keywords_dict: dict[str, str],
        fnc: Callable[[str, str], list[str]],
        chosen_nums: list[int],
    ) -> tuple[list[str], list[str]]:
        counter = 1
        chosen_keywords_lst = []
        chosen_replacements_lst = []
        for keyword in keywords_lst:
            if counter <= max(chosen_nums):
                keyword_pos = keywords_dict[keyword][0].lower()
                if keyword_pos == "j":
                    keyword_pos = "a"
                candidates = fnc(keyword, keyword_pos)
                if candidates:
                    counter += 1
                    chosen_keywords_lst.append(keyword)
                    chosen_replacement = self.rng.choice(candidates)
                    chosen_replacements_lst.append(chosen_replacement)
            else:
                return chosen_keywords_lst, chosen_replacements_lst
        return chosen_keywords_lst, chosen_replacements_lst

    def single_prompt_wordnet(self, prompt: str, nums_lst: list[int]) -> str:
        original_prompt = prompt
        synonyms_prompt_str = ""
        keywords_dict = self.extract_keywords_and_pos(prompt)

        if not keywords_dict:
            return ""

        keywords_lst = list(keywords_dict.keys())
        prompt_synonym = original_prompt

        chosen_keywords, chosen_synonyms = self.single_prompt_helper(
            keywords_lst,
            keywords_dict,
            self.get_new_keyword,
            nums_lst,
        )
        counter = 1

        for chosen_word, chosen_synonym in zip(chosen_keywords, chosen_synonyms, strict=False):
            prompt_synonym = re.sub(rf"\b{chosen_word}\b", chosen_synonym, prompt_synonym)
            if counter in nums_lst:
                synonyms_prompt_str += re.sub("_", " ", prompt_synonym) + " "
            counter += 1

        return synonyms_prompt_str.strip()

    def random_aug(self, sentence: str, alpha: float, choice: str) -> str:
        words = sentence.split()
        words = [word for word in words if word]
        num_words = len(words)
        n1 = max(1, int(alpha * num_words))

        if choice == "insertion":
            a_words = self.random_insertion(words, n1)
        elif choice == "swap":
            a_words = self.random_swap(words, n1)
        elif choice == "deletion":
            a_words = self.random_deletion(words, alpha)
        elif choice == "kreplacement":
            return self.single_prompt_wordnet(sentence, [3])
        else:
            raise ValueError("Invalid choice. Choose from 'insertion','kreplacement', 'swap', or 'deletion'.")

        result_sentence = " ".join(a_words)
        return re.sub(" +", " ", result_sentence).strip()
