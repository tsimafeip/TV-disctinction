from tv_detector import SimpleTVDetector
from conll_tv_detector import ConllTVDetector

from helper import reformat_deixis_files, report_metrics, prepend_label_to_all_lines, create_tv_subcorpus, \
    compare_tv_detectors

if __name__ == '__main__':
    report_metrics(translation_filename='../translations/base_model_no_labels',
                   reference_filename='../data/deixis_test_ru')
    report_metrics(translation_filename='../translations/tv_model_no_labels',
                   reference_filename='../data/deixis_test_ru')
    report_metrics(translation_filename='../translations/tv_model_oracle_labels',
                   reference_filename='../data/deixis_test_ru')
    report_metrics(translation_filename='../translations/tv_model_t_labels',
                   reference_filename='../data/deixis_test_ru')
    report_metrics(translation_filename='../translations/tv_model_v_labels',
                   reference_filename='../data/deixis_test_ru')

    # prepend_label_to_all_lines(source_filename='../data/deixis_test_en', label='<T>',
    #                            target_filename='../data/deixis_test_en.t')
    # prepend_label_to_all_lines(source_filename='../data/deixis_test_en', label='<V>',
    #                            target_filename='../data/deixis_test_en.v')

    # create_tv_subcorpus(source_ru_filepath='corpus.en_ru.1m.ru', source_en_filepath='corpus.en_ru.1m.en',
    #                     target_ru_filepath='yandex_subcorpus_ru.txt', target_en_filepath='yandex_subcorpus_en.txt')
    #
    # reformat_deixis_files('./../data/good-translation-wrong-in-context-deixis/',
    #                       en_res_filepath='../data/deixis_test_en', ru_res_filepath='../data/deixis_test_ru')

    compare_tv_detectors('../translations/base_model_no_labels')

    simple_detector = SimpleTVDetector()
    simple_detector.detect_t_v_labels(filename='../translations/base_model_no_labels')
    # simple_detector.mark_tv_sentences(source_filename='../data/deixis_test_en',
    #                                   target_filename='../data/deixis_test_ru')

    conll_detector = ConllTVDetector()
    conll_detector.detect_t_v_labels(filename='../translations/base_model_no_labels.conll')
    # conll_detector.mark_tv_sentences(source_filename='../data/deixis_test_en',
    #                                  target_filename='./../data/deixis_test_ru.conll')
