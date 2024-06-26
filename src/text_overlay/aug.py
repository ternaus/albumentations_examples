import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

import albumentations as A
import numpy as np
from albumentations.core.bbox_utils import check_bbox, denormalize_bbox
from nltk.corpus import wordnet
from nltk.tag.stanford import StanfordPOSTagger
from PIL import ImageFont
from rake_nltk import Rake

from src.text_overlay.utils import render_text

seed = 42

rng = np.random.default_rng(seed)  # Initialize the random generator with an optional seed


def get_stopwords(path: Path) -> list[str]:
    with path.open() as f:
        stopwords = f.readlines()
    return [x.strip() for x in stopwords]


def random_deletion(words: list[str], alpha: float) -> list[str]:
    if len(words) <= 1:
        return words
    new_words = [word for word in words if rng.uniform() > alpha]
    return new_words if new_words else [rng.choice(words)]


def swap_word(new_words: list[str]) -> list[str]:
    if len(new_words) < 2:
        return new_words
    random_idx_1 = rng.integers(len(new_words))
    counter = 0
    while True:
        random_idx_2 = rng.integers(len(new_words))
        if random_idx_1 != random_idx_2:
            new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
            break
        counter += 1
        if counter > 3:
            break
    return new_words


def random_swap(words: list[str], n: int) -> list[str]:
    new_words = words.copy()
    for _ in range(n):
        new_words = swap_word(new_words)
    return new_words


def get_synonyms(word: str) -> list[str]:
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonyms.add("".join([char for char in synonym if char.isalpha() or char == " "]))
    synonyms.discard(word)
    return list(synonyms)


def random_insertion(words: list[str], n: int, stopwords: list[str]) -> list[str]:
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words, stopwords)
    return new_words


def add_word(new_words: list[str], stopwords: list[str]) -> None:
    non_stopwords = [word for word in new_words if word not in stopwords]
    if not non_stopwords:
        return
    random_word = rng.choice(non_stopwords)
    synonyms = get_synonyms(random_word)
    if synonyms:
        random_synonym = rng.choice(synonyms)
        random_idx = rng.integers(len(new_words))
        new_words.insert(random_idx, random_synonym)


def extract_keywords_and_pos(post_tagger: StanfordPOSTagger, rake: Rake, prompt: str) -> dict[str, str]:
    pos_dict = {}
    try:
        tagged_prompt = post_tagger.tag(prompt.split())
    except:
        return {}
    for word, pos in tagged_prompt:
        pos_dict[word] = pos
    keywords_dict = {}
    keywords = rake.run(prompt)
    for pair in keywords:
        words = pair[0].split()
        for word in words:
            if word in pos_dict:
                keywords_dict[word] = pos_dict[word]
    return keywords_dict


def get_new_keyword(word: str, pos: str) -> list[str]:
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
                chosen_replacement = rng.choice(candidates)
                chosen_replacements_lst.append(chosen_replacement)
        else:
            return chosen_keywords_lst, chosen_replacements_lst
    return chosen_keywords_lst, chosen_replacements_lst


def single_prompt_wordnet(post_tagger: StanfordPOSTagger, rake: Rake, prompt: str, nums_lst: list[int]) -> str:
    original_prompt = prompt
    synonyms_prompt_str = ""
    keywords_dict = extract_keywords_and_pos(post_tagger, rake, prompt)

    if not keywords_dict:
        return ""

    keywords_lst = list(keywords_dict.keys())
    prompt_synonym = original_prompt

    chosen_keywords, chosen_synonyms = single_prompt_helper(
        keywords_lst,
        keywords_dict,
        get_new_keyword,
        nums_lst,
    )
    counter = 1

    for chosen_word, chosen_synonym in zip(chosen_keywords, chosen_synonyms, strict=False):
        prompt_synonym = re.sub(rf"\b{chosen_word}\b", chosen_synonym, prompt_synonym)
        if counter in nums_lst:
            synonyms_prompt_str += re.sub("_", " ", prompt_synonym) + " "
        counter += 1

    return synonyms_prompt_str.strip()


def copy_and_paste_blend(
    base_image: np.ndarray,
    overlay_image: np.ndarray,
    offset: tuple[int, int],
) -> np.ndarray:
    y_offset, x_offset = offset
    blended_image = base_image.copy()
    height, width = overlay_image.shape[:2]

    blended_image[y_offset : y_offset + height, x_offset : x_offset + width] = overlay_image

    return blended_image


class TextAugmenter(A.ImageOnlyTransform):
    def __init__(
        self,
        font_path: Path,
        augmentations: tuple[str, ...] = ("insertion", "swap", "deletion", "kreplacement"),
        fraction_range: tuple[float, float] = (0.1, 0.9),
        stopwords_path: Path | None = None,
        pos_model_path: Path | None = None,
        pos_jar_path: Path | None = None,
        metadata_key: str = "textaug_metadata",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.metadata_key = metadata_key
        self.font_path = font_path
        self.fraction_range = fraction_range
        temp_aug_list = list(augmentations)

        if stopwords_path is None:
            self.stopwords = []
            self.rake = None
            temp_aug_list = [item for item in temp_aug_list if item not in {"kreplacement", "insertion"}]
        else:
            self.stopwords = get_stopwords(stopwords_path)
            self.rake = Rake(stopwords_path)

        if pos_model_path is None or pos_jar_path is None:
            self.post_tagger = None
            self.rake = None
        else:
            pos_model_path = pos_model_path.absolute()
            pos_jar_path = pos_jar_path.absolute()
            self.post_tagger = StanfordPOSTagger(
                model_filename=pos_model_path,
                path_to_jar=pos_jar_path,
                java_options="-mx4000m",
            )

        self.augmentations = temp_aug_list

    @property
    def targets_as_params(self) -> list[str]:
        return [self.metadata_key, "image"]

    def random_aug(
        self,
        sentence: str,
        alpha: float,
        choice: Literal["insertion", "swap", "deletion", "kreplacement"],
    ) -> str:
        words = sentence.strip().split()
        words = [word for word in words if word]
        num_words = len(words)
        n1 = max(1, int(alpha * num_words))

        if choice == "insertion":
            a_words = random_insertion(words, n1, self.stopwords)
            if len(a_words) == 0 or " ".join(a_words) == sentence:
                result_sentence = ""
            else:
                result_sentence = " ".join(a_words)
                result_sentence = re.sub(" +", " ", result_sentence)
        elif choice == "swap":
            a_words = random_swap(words, n1)
            result_sentence = "" if len(a_words) == 0 or " ".join(a_words) == sentence else " ".join(a_words)
        elif choice == "deletion":
            a_words = random_deletion(words, alpha)
            result_sentence = "" if len(a_words) == 0 or " ".join(a_words) == sentence else " ".join(a_words)
        elif choice == "kreplacement":
            return single_prompt_wordnet(self.post_tagger, self.rake, sentence, [3])
        else:
            raise ValueError("Invalid choice. Choose from 'insertion','kreplacement', 'swap', or 'deletion'.")

        return result_sentence

    def preprocess_metadata(self, img_shape: tuple[int, ...], bbox: tuple[float, ...], text: str) -> dict[str, Any]:
        image_height, image_width = img_shape[:2]

        check_bbox(bbox)
        denormalized_bbox = denormalize_bbox(bbox[:4], rows=image_height, cols=image_width)

        x_min, y_min, x_max, y_max = (int(x) for x in denormalized_bbox[:4])

        bbox_height = y_max - y_min
        bbox_width = x_max - x_min

        font = ImageFont.truetype(self.font_path, int(0.90 * bbox_height))

        if len(text) <= 50:
            filtered_list = [item for item in self.augmentations if item != "kreplacement"]
            augmentation = random.choice(filtered_list)
        else:
            augmentation = random.choice(self.augmentations)

        augmented_text = self.random_aug(
            text,
            0.5,
            cast(Literal["insertion", "swap", "deletion", "kreplacement"], augmentation),
        )

        overlay_image = render_text((bbox_height, bbox_width), augmented_text, font)

        offset = (y_min, x_min)

        return {
            "overlay_image": overlay_image,
            "offset": offset,
        }

    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        metadata = params[self.metadata_key]
        img_shape = params["image"].shape
        bboxes = metadata["bboxes"]
        texts = metadata["texts"]
        fraction = random.uniform(*self.fraction_range)

        num_texts = len(texts)

        num_lines_to_modify = int(len(texts) * fraction)

        bbox_indices_to_update = rng.choice(range(num_texts), num_lines_to_modify)

        overlay_data = [
            self.preprocess_metadata(img_shape, bboxes[index], texts[index]) for index in bbox_indices_to_update
        ]

        return {
            "overlay_data": overlay_data,
        }

    def apply(
        self,
        img: np.ndarray,
        overlay_data: list[dict[str, Any]],
        **params: Any,
    ) -> np.ndarray:
        for data in overlay_data:
            overlay_image = data["overlay_image"]
            offset = data["offset"]
            img = copy_and_paste_blend(img, overlay_image, offset=offset)
        return img
