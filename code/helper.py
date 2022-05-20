import os
from typing import Tuple, List
from sacrebleu import corpus_bleu

from tv_detector import SimpleTVDetector
from conll_tv_detector import ConllTVDetector

DEIXIS_EOS_SEPARATOR = '_eos'


def compare_tv_detectors(source_filepath: str):
    """Reports accuracy of token-based T/V detector compared to the grammar-based one.

    Parameters
    ----------
    source_filepath: str
        Path to the file with source Russian sentences.
    """
    simple_detector = SimpleTVDetector()
    token_based_tv_list = simple_detector.detect_t_v_labels(filename=source_filepath)

    simple_t_num = sum(t_item for t_item, v_item in token_based_tv_list)
    simple_v_num = sum(v_item for t_item, v_item in token_based_tv_list)

    conll_detector = ConllTVDetector()
    conll_filepath = source_filepath + '.conll'
    conll_tv_list = conll_detector.detect_t_v_labels(filename=conll_filepath)

    conll_t_num = sum(t_item for t_item, v_item in conll_tv_list)
    conll_v_num = sum(v_item for t_item, v_item in conll_tv_list)

    percent = (simple_t_num + simple_v_num) / (conll_v_num + conll_t_num)

    print("Percentage of T/V sentences found by token-based detector comparing to grammar-based:", percent)


def prepend_label_to_all_lines(source_filename: str, label: str, target_filename: str) -> None:
    """Prepends specific label to each line from source_file and stores it in target file.

    Parameters
    ----------
    source_filename: str
        Name of the file with source sentences.
    label
        Special token to prepend to each line. For this project we use <T> and <V>.
    target_filename: str
        Name of the file to store labeled sentences.
    """

    with open(source_filename) as input_f, open(target_filename, 'w') as output_f:
        labeled_lines = [label + ' ' + line + '\n' for line in input_f.read().splitlines()]
        output_f.writelines(labeled_lines)


def _extract_sentences_en_ru_from_deixis_files(path_to_files: str, ru_filename: str, en_filename: str) \
        -> Tuple[List[str], List[str]]:
    """Splits and deduplicates deixis-related files from Voita et. al. (2019).

    Link to the source:
    https://github.com/lena-voita/good-translation-wrong-in-context/tree/master/consistency_testsets/scoring_data

    Parameters
    ----------
    path_to_files: str
        Path to source directory.
    ru_filename: str
        Name of file in Russian.
    en_filename: str
        Name of file in English.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of parallel English and Russian sentences.
    """
    ru_set = set()
    ru_sentences = []
    en_sentences = []
    with open(os.path.join(path_to_files, ru_filename)) as ru_file, \
            open(os.path.join(path_to_files, en_filename)) as en_file:

        for (ru_line, en_line) in zip(ru_file, en_file):
            ru_sentences_in_line = ru_line.strip().split(DEIXIS_EOS_SEPARATOR)
            en_sentences_in_line = en_line.strip().split(DEIXIS_EOS_SEPARATOR)
            assert len(ru_sentences_in_line) == len(en_sentences_in_line)
            for ru_sentence, en_sentence in zip(ru_sentences_in_line, en_sentences_in_line):
                # duplicate translation
                if ru_sentence in ru_set:
                    continue
                ru_sentences.append(ru_sentence.strip())
                en_sentences.append(en_sentence.strip())
                ru_set.add(ru_sentence)

    assert len(en_sentences) == len(ru_sentences)

    return en_sentences, ru_sentences


def reformat_deixis_files(
        path_to_files: str, en_res_filepath: str = "deixis_test_en", ru_res_filepath: str = "deixis_test_ru",
) -> None:
    """Reformats and merges deixis-related files from Voita et.al. (2019) and writes resulting new files.

    Link to the source:
    https://github.com/lena-voita/good-translation-wrong-in-context/tree/master/consistency_testsets/scoring_data

    Parameters
    ----------
    path_to_files: str
        Path to source directory.
    en_res_filepath: str
        Name of the resulting file in Russian.
    ru_res_filepath: str
        Name of the resulting file in English.
    """

    # extract sentences from dev files
    dev_en_sentences, dev_ru_sentences = _extract_sentences_en_ru_from_deixis_files(
        path_to_files, ru_filename='deixis_dev.dst', en_filename='deixis_dev.src',
    )

    # extract sentences from test files
    test_en_sentences, test_ru_sentences = _extract_sentences_en_ru_from_deixis_files(
        path_to_files, ru_filename='deixis_test.dst', en_filename='deixis_test.src',
    )

    en_sentences = test_en_sentences + dev_en_sentences
    ru_sentences = test_ru_sentences + dev_ru_sentences

    # there are two sentences that translate note signs in English to '.' in Russian
    # some models are translating it as an empty string, which causes problems on evaluation
    # it is some kind of hardcode fix, sorry
    skip_indices = {i for i, ru_sent in enumerate(ru_sentences) if ru_sent == '.'}

    with open(en_res_filepath, 'w', encoding='utf-8') as en_file, \
            open(ru_res_filepath, 'w', encoding='utf-8') as ru_file:
        en_file.writelines([en_sent + '\n' for i, en_sent in enumerate(en_sentences) if i not in skip_indices])
        ru_file.writelines([ru_sent + '\n' for i, ru_sent in enumerate(ru_sentences) if i not in skip_indices])


def report_metrics(translation_filename: str, reference_filename: str) -> None:
    """Reports quality of translation in terms of BLEU, BLEURT and T/V labeling.

    Requires BLEURT and CoNLL files to be created in advance
    by running `code/notebooks/bleurt_evaluation.ipynb` and `code/notebooks/parse_file_to_conll.ipynb`.

    Parameters
    ----------
    translation_filename: str
        Name of the file with candidate translations.
    reference_filename: str
        Name of the file with reference translations.

    Raises
    ------
    FileNotFoundError
        If any of BLEURT or CoNLL files are not found.
    """

    translation_conll_filename = translation_filename + '.conll'
    translation_bleurt_filename = translation_filename + '.bleurt'
    if not os.path.isfile(translation_bleurt_filename):
        raise FileNotFoundError(f"Cannot find file with calculated BLEURT scores: {translation_bleurt_filename}."
                                f"Please, run `code/notebooks/bleurt_evaluation.ipynb` first.")

    if not os.path.isfile(translation_conll_filename):
        raise FileNotFoundError(f"Cannot find file of translations parsed to CoNLLL: {translation_conll_filename}."
                                f"Please, run `code/notebooks/parse_file_to_conll.ipynb` first.")

    print(f"Reporting metrics for the '{translation_filename}' ...")
    report_tv_in_file(translation_conll_filename)
    bleu_score = get_bleu(translation_filename, reference_filename)
    bleurt_score = get_bleurt_score_for_corpus(translation_bleurt_filename)

    print('BLEU score: ', bleu_score)
    print('BLEURT score: ', bleurt_score)
    print()


def report_tv_in_file(conll_filename: str) -> None:
    """Calculates number of T/V sentences using grammar-based approach.

    Parameters
    ----------
    conll_filename: str
        Name of the file with sentences parsed to CoNLL format.
    """

    conll_detector = ConllTVDetector()
    conll_detector.detect_t_v_labels(filename=conll_filename)


def get_bleu(translation_filename: str, reference_filename: str) -> float:
    """Calculates BLEU metric for provided files.

    Parameters
    ----------
    translation_filename: str
        Name of the file with candidate translations.
    reference_filename: str
        Name of the file with reference translations.

    Returns
    -------
    float
        BLEU score.
    """
    with open(translation_filename, 'r', encoding="utf-8") as prediction_f, \
            open(reference_filename, 'r', encoding="utf-8") as reference_f:
        real = reference_f.readlines()
        prediction = prediction_f.readlines()

    return corpus_bleu(prediction, [real]).score


def get_bleurt_score_for_corpus(bleurt_filename: str) -> float:
    """Reports BLEURT score based on precalculated file produced by `code/notebooks/bleurt_evaluation.ipynb`.

    Parameters
    ----------
    bleurt_filename: str
        Name of the file with BLEURT scores for each sentence pair.

    Returns
    -------
    float
        Mean of all sentence-level BLEURT scores multiplied by 100.
    """

    with open(bleurt_filename) as input_f:
        scores = list(map(float, input_f.read().splitlines()))
        return 100 * (sum(scores) / len(scores))


def create_tv_subcorpus(source_ru_filepath: str, source_en_filepath: str,
                        target_ru_filepath: str, target_en_filepath: str) -> None:
    """Creates TV-balanced sub-corpus from Yandex 1m corpus.

    Parameters
    ----------
    source_ru_filepath: str
        Path to the file with 1 million of Russian sentences.
    source_en_filepath: str
        Path to the file with 1 million of English sentences.
    target_ru_filepath: str
        Path to the file with resulting Russian sentences.
    target_en_filepath: str
        Path to the file with resulting English sentences.
    """

    ru_sentences = []
    en_sentences = []
    n_sentences_num = v_sentences_num = t_sentences_num = 0

    # limits are empirical, they are set to create a kind of balanced dataset
    V_LIMIT = 22000
    N_LIMIT = 100000

    detector = SimpleTVDetector()
    t_v_found = detector.detect_t_v_labels(filename=source_ru_filepath)

    with open(source_en_filepath, 'r', encoding='utf-8') as ru_file, \
            open(source_en_filepath, 'r', encoding='utf-8') as en_file:

        for (t_found, v_found), (ru_line, en_line) in zip(t_v_found, zip(ru_file, en_file)):

            if not t_found and len(ru_line) > 100:
                continue

            n_found = (not t_found and not v_found)

            n_sentences_num += n_found
            v_sentences_num += v_found
            t_sentences_num += t_found

            if (v_found and v_sentences_num > V_LIMIT) \
                    or (n_found and n_sentences_num > N_LIMIT):
                continue

            ru_sentences.append(ru_line)
            en_sentences.append(en_line)

    with open(target_ru_filepath, 'w', encoding='utf-8') as ru_file, \
            open(target_en_filepath, 'w', encoding='utf-8') as en_file:

        ru_file.writelines(ru_sentences)
        en_file.writelines(en_sentences)
