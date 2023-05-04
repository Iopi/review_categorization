import os

import constants
import numpy as np
from view import app_output


def cosine_similarity(vec_a, vec_b):
    """
    Calculates cosine similarity
    :param vec_a: first vector
    :param vec_b: second vector
    :return:  calculated cosine similarity
    """
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def eval_similarity(target_model, source_model, target_lang, source_lang, trans_matrix):
    """
    Accuracy calculation of transformation matrix
    :param target_model: target vector model
    :param source_model: source vector model
    :param target_lang: target language
    :param source_lang: source language
    :param trans_matrix: transformation matrix
    :return:
    """
    filename = constants.DICT_FOLDER + f"{target_lang}-{source_lang}.txt"
    if not os.path.exists(filename):
        app_output.exception(f"Dictionary {filename} not found.")

    sum = 0
    i = 0
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            wt, ws = line.split()
            try:
                vec_t = target_model[wt]
                vec_s = source_model[ws]
            except:
                continue

            trans_vec_s = np.dot(trans_matrix, vec_s)
            simi = cosine_similarity(vec_t, trans_vec_s)
            i += 1
            sum += simi

    app_output.output(f"sum - {sum / i}")


def compute_transform_matrix_regression(target_model, source_model, dic_filename):
    """
    Calculation of transformation matrix by regression method
    :param target_model: target vector model
    :param source_model: source vector model
    :param dic_filename: dictionary filename
    :return: calculated transformation matrix
    """
    # read dictionary <target_word translate_of_target_word>
    with open(dic_filename, encoding="utf-8") as f:
        matrix_t = None
        matrix_s = None
        for line in f:
            line = line.rstrip()
            wt, ws = line.split()
            # get words' vectors
            try:
                vec_t = target_model[wt]
                vec_s = source_model[ws]
            except:
                continue

            # matrix assembly
            if matrix_t is None:
                matrix_t = vec_t
                matrix_s = vec_s
            else:
                matrix_t = np.vstack((matrix_t, vec_t))
                matrix_s = np.vstack((matrix_s, vec_s))

        # transposition of the target matrix
        matrix_s_T = matrix_s.T

        trans_matrix_1 = np.dot(matrix_s_T, matrix_s)
        trans_matrix_2 = np.dot(matrix_s_T, matrix_t)
        trans_matrix_1 = np.linalg.inv(trans_matrix_1)

        trans_matrix = np.dot(trans_matrix_1, trans_matrix_2)

    return trans_matrix.T


def compute_transform_matrix_orthogonal(target_model, source_model, dic_filename):
    """
    Calculation of transformation matrix by orthogonal method
    :param target_model: target vector model
    :param source_model: source vector model
    :param dic_filename: dictionary filename
    :return: calculated transformation matrix
    """
    # read dictionary <target_word translate_of_target_word>
    with open(dic_filename, encoding="utf-8") as f:
        matrix_t = None
        matrix_s = None
        for line in f:
            line = line.rstrip()
            wt, ws = line.split()
            # get words' vectors
            try:
                vec_t = target_model[wt]
                vec_s = source_model[ws]
            except:
                continue

            # matrix assembly
            if matrix_t is None:
                matrix_t = vec_t
                matrix_s = vec_s
            else:
                matrix_t = np.vstack((matrix_t, vec_t))
                matrix_s = np.vstack((matrix_s, vec_s))
        # transposition of the target matrix
        matrix_t_T = matrix_t.T
        # multiplication of transposed target matrix and source matrix
        X = np.dot(matrix_t_T, matrix_s)
        # Singular Value Decomposition
        U, s, V_T = np.linalg.svd(X)
        V = V_T.T[:, :len(s)]
        U_T = U.T
        trans_matrix = np.dot(V, U_T)
    return trans_matrix.T


def get_trans_matrix(vec_model_train, vec_model_test, target_lang, source_lang, trans_method, dic_filename):
    """
    Calculation of transformation matrix
    :param vec_model_train: train vector model
    :param vec_model_test: test vector model
    :param target_lang: target language
    :param source_lang: source language
    :param trans_method: transformation method
    :param dic_filename: dictionary filename
    :return: Calculated transformation matrix
    """
    if trans_method == "orto":
        trans_matrix = compute_transform_matrix_orthogonal(vec_model_train, vec_model_test, dic_filename)
    elif trans_method == "regr":
        trans_matrix = compute_transform_matrix_regression(vec_model_train, vec_model_test, dic_filename)
    else:
        app_output.exception(f"Unknown transform matrix method {trans_method}")

    # calculate accuracy of transformation matrix
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)

    return trans_matrix
