from typing import Tuple, List, Optional

import conllu
from conllu import parse_incr as conll_parse_incr, parse as conll_parse

from tv_detector import TVDetector


class ConllTVDetector(TVDetector):
    """Implementation of T/V Detector using grammar-based approach and set of heuristics."""

    def detect_t_v_labels(
            self, lines: Optional[List[str]] = None, filename: Optional[str] = None,
    ) -> List[Tuple[bool, bool]]:
        """Detects T/V labels using grammar-based information.

        Accepts either list of strings in CoNLL format or path to source file with CoNLL data.

        Parameters
        ----------
        lines: List[str], optional
            Sentences parsed to CoNLL format.
        filename: str, optional
            Name of file, which holds sentences parsed to CoNLL format.

        Returns
        -------
        List[Tuple[bool, bool]]
            Returns list of tuples of the (bool, bool) format with meaning (t_label, v_label)
            for the corresponding sentence from the source list/file.

        Raises
        ------
        RuntimeError
            If both source lines list and filename provided. Only one option can be specified.
        conllu.exceptions.ParseException
            If provided file or sentences are not in CoNLL format.
        """

        if filename is None and lines is None:
            raise RuntimeError('Error occured on T/V labels detection. '
                               'Either source file or list of sentences have to be provided.')

        t_v_list = []

        if filename:
            data_file = open(filename, "r", encoding="utf-8")
            for tokenlist in conll_parse_incr(data_file):
                t_v_list.append(self._detect_t_v_from_conll(tokenlist))
        else:
            for line in lines:
                for tokenlist in conll_parse(line):
                    t_v_list.append(self._detect_t_v_from_conll(tokenlist))

        t_sentences_num = sum(t_item for t_item, v_item in t_v_list)
        v_sentences_num = sum(v_item for t_item, v_item in t_v_list)
        print(f'Neutral sentences: {len(t_v_list) - v_sentences_num - t_sentences_num}. '
              f'V sentences found: {v_sentences_num}. '
              f'T sentences found: {t_sentences_num}. ')

        return t_v_list

    @classmethod
    def _check_token_for_v(cls, parsed_token: conllu.models.Token) -> bool:
        """Checks if passed token is V-specific or not.

        We check V-form lemma of the specific Russian pronoun (polite you) and determiner (polite your).
        Also, we look for verbs of plural number and second person.

        Parameters
        ----------
        parsed_token: conllu.models.Token
            CoNLL token with morphological information.

        Returns
        -------
        bool
            flag determining whether we found V-specific token or not.
        """

        return (parsed_token['upos'] == 'PRON' and parsed_token['lemma'] == 'вы') or \
               (parsed_token['upos'] == 'DET' and parsed_token['lemma'] == 'ваш') or \
               (parsed_token['upos'] == 'VERB' and
                parsed_token['feats'].get('Number') == 'Plur' and
                parsed_token['feats'].get('Person') == '2')

    @classmethod
    def _check_token_for_t(cls, parsed_token: conllu.models.Token) -> bool:
        """Checks if passed token is T-specific or not.

        We check T-form lemma of the specific Russian pronoun (informal you) and determiner (informal your).
        Also, we look for verbs of single number and second person.

        Parameters
        ----------
        parsed_token: conllu.models.Token
            CoNLL token with morphological information.

        Returns
        -------
        bool
            flag determining whether we found T-specific token or not.
        """
        return (parsed_token['upos'] == 'PRON' and parsed_token['lemma'] == 'ты') \
               or (parsed_token['upos'] == 'DET' and parsed_token['lemma'] == 'твой') \
               or (parsed_token['upos'] == 'VERB'
                   and parsed_token['feats'].get('Number') == 'Sing'
                   and parsed_token['feats'].get('Person') == '2')

    @classmethod
    def _detect_t_v_from_conll(cls, conll_token_list: conllu.models.TokenList) -> Tuple[bool, bool]:
        """Performs grammar-based T/V detection.

        Iterates by token list in CoNLLL format and looks for T-specific or V-specific tokens,
        which we determine using set of grammar-based heuristics.

        If both T/V-specific found, then sentences is marked as neutral.

        Parameters
        ----------
        conll_token_list: conllu.models.TokenList
            Source sentence parsed to token list in CoNLL format.

        Returns
        -------
        Tuple[bool, bool]
            Tuple of the (bool, bool) format with meaning (t_label, v_label).
        """
        v_token_met = t_token_met = False
        for parsed_token in conll_token_list:
            v_token_met |= cls._check_token_for_v(parsed_token)
            t_token_met |= cls._check_token_for_t(parsed_token)

        t_sentence_found = t_token_met and not v_token_met
        v_sentence_found = v_token_met and not t_token_met

        return t_sentence_found, v_sentence_found
