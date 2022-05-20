from abc import ABC, abstractmethod
from typing import Tuple, Set, List, Optional

from tqdm import tqdm

SPECIAL_T_TOKEN = '<T>'
SPECIAL_V_TOKEN = '<V>'


class TVDetector(ABC):
    """
    Abstract base class for T/V detector.

    Methods
    -------
    detect_t_v_labels(lines: Optional[List[str]] = None, filename: Optional[str] = None)
        Abstract method, that must be implemented in all inherited classes.
        Accepts either list of strings or path to source file to perform T/V detection.
        Return list of tuples of the (bool, bool) format with meaning (t_label, v_label)
        for the corresponding sentence from the source list/file.

    mark_tv_sentences(source_filename: str, target_filename: str)
        Marks tv_sentences from source_filename with T/V labels based on information from target.
        Writes resulting file with the same name as source_filename and '.tv' extension.
    """

    @abstractmethod
    def detect_t_v_labels(
            self, lines: Optional[List[str]] = None, filename: Optional[str] = None,
    ) -> List[Tuple[bool, bool]]:
        """ Detects T/V labels Accepts either list of strings or path to source file to perform T/V detection.

        Parameters
        ----------
        lines: List[str], optional
            Sentences to label with T/V.
        filename: str, optional
            Name of file, which holds sentences to label with T/V.

        Returns
        -------
        List[Tuple[bool, bool]]
            Returns list of tuples of the (bool, bool) format with meaning (t_label, v_label)
            for the corresponding sentence from the source list/file.
        """

        pass

    def mark_tv_sentences(self, source_filename: str, target_filename: str) -> str:
        """Prepends T/V labels to source sentences based on information from target file.

        Parameters
        ----------
        source_filename: str
            Name of source file, which we will use to extract sentences and prepend T/V tokens to them.
        target_filename: str
            Name of target file, which we will use only to extract T/V labeling information to update source data.

        Returns
        -------
        str
            Name of resulting file with labeled T/V sentences.
        """
        labeled_source_sentences = []

        with open(source_filename, 'r', encoding='utf-8') as en_file:

            t_v_detected = self.detect_t_v_labels(filename=target_filename)

            for source_line, (t_found, v_found) in tqdm(zip(en_file, t_v_detected)):

                if t_found:
                    labeled_source_sentences.append(SPECIAL_T_TOKEN + ' ' + source_line)
                elif v_found:
                    labeled_source_sentences.append(SPECIAL_V_TOKEN + ' ' + source_line)
                else:
                    labeled_source_sentences.append(source_line)

        tv_source_filename = source_filename + '.tv'
        print(f'\nWriting sentences with T/V labels to the file {tv_source_filename}...')

        with open(tv_source_filename, 'w', encoding='utf-8') as tv_en_file:
            tv_en_file.writelines(labeled_source_sentences)

        return tv_source_filename


class SimpleTVDetector(TVDetector):
    """Implementation of T/V Detector using simple token-based matching approach."""

    def __init__(self):
        """Initialises SimpleTVDetector object."""
        t_words, v_words = self._get_russian_T_V_words()
        self.russian_T_words = t_words
        self.russian_V_words = v_words

    def detect_t_v_labels(
            self, lines: Optional[List[str]] = None, filename: Optional[str] = None,
    ) -> List[Tuple[bool, bool]]:
        """Detects T/V labels using token-based word matching.

        Accepts either list of strings or path to source file to perform T/V detection.

        Parameters
        ----------
        lines: List[str], optional
            Sentences to label with T/V.
        filename: str, optional
            Name of file, which holds sentences to label with T/V.

        Returns
        -------
        List[Tuple[bool, bool]]
            Returns list of tuples of the (bool, bool) format with meaning (t_label, v_label)
            for the corresponding sentence from the source list/file.

        Raises
        ------
        RuntimeError
            If both source lines list and filename provided. Only one option can be specified.
        """

        if filename is None and lines is None:
            raise RuntimeError('Error occured on T/V labels detection. '
                               'Either source file or list of sentences have to be provided.')

        t_v_list = []

        if filename:
            data_file = open(filename, "r", encoding="utf-8")
            lines = data_file.read().splitlines()

        for line in lines:
            t_v_list.append(self._token_based_t_v_labels_detection(line))

        t_sentences_num = sum(t_item for t_item, v_item in t_v_list)
        v_sentences_num = sum(v_item for t_item, v_item in t_v_list)
        print(f'Neutral sentences: {len(t_v_list) - v_sentences_num - t_sentences_num}. '
              f'V sentences found: {v_sentences_num}. '
              f'T sentences found: {t_sentences_num}.')

        return t_v_list

    @classmethod
    def _get_russian_T_V_words(cls) -> Tuple[Set[str], Set[str]]:
        """Returns tuple of Russian T/V specific words, which were collected manually by the author.

        Returns
        -------
        Tuple[Set[str], Set[str]]
            tuple of T-specific and V-specific Russian words.
        """

        russian_T_tokens = {
            'ты', 'тебя', 'тебе', 'тобою', 'тобой',
            'твой', 'твоего', 'твоему', 'твоим', 'твоём', 'твоeм',
            'твоё', 'твое',
            'твоя', 'твоей', 'твою', 'твоею', 'твоей',
            'твои', 'твоих', 'твоим', 'твоими',
        }
        russian_T_tokens.update({token[0].upper() + token[1:] for token in russian_T_tokens})

        russian_V_tokens = {
            'вы', 'вас', 'вам', 'вами',
            'ваш', 'вашего', 'вашему', 'вашего', 'вашим', 'вашем',
            'ваше',
            'ваша', 'вашей', 'вашу', 'вашею',
            'ваши', 'ваших', 'вашими', 'ваших',
        }
        russian_V_tokens.update({token[0].upper() + token[1:] for token in russian_V_tokens})

        return russian_T_tokens, russian_V_tokens

    def _token_based_t_v_labels_detection(self, line: str) -> Tuple[bool, bool]:
        """Performs token-based T/V detection.

        Splits provided line by space and doing lookup of T/V-specific tokens.
        If both T/V-specific found, then sentences is marked as neutral.

        Parameters
        ----------
        line: str
            source sentence to label with T/V.

        Returns
        -------
        Tuple[bool, bool]
            tuple of the (bool, bool) format with meaning (t_label, v_label).
        """

        tokens = set(line.strip().split())
        t_token_met = bool(self.russian_T_words & tokens)
        v_token_met = bool(self.russian_V_words & tokens)

        t_sentence_found = t_token_met and not v_token_met
        v_sentence_found = v_token_met and not t_token_met

        return t_sentence_found, v_sentence_found