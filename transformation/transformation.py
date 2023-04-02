import os

import numpy as np

import constants
import util


def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


def eval_similarity(target_model, source_model, target_lang, source_lang, trans_matrix):
    filename = constants.DICT_FOLDER + f"{target_lang}-{source_lang}_muj.txt"
    if not os.path.exists(filename):
        util.exception(f"Dictionary {filename} not found.")

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
            # print(line)
            # print(vec_t[:10])
            # print(trans_vec_s[:10])  # hodnoty vektoru nejsou vubec mezi -1:1 pro regression
            # print(vec_s[:10])
            # print(cosine_similarity(vec_t, vec_s))

            # print(simi)
            # print()

    print(f"sum - {sum / i}")


def compute_transform_matrix_regression(target_model, source_model, filename):
    with open(filename, encoding="utf-8") as f:
        matrix_t = None
        matrix_s = None
        for line in f:
            line = line.rstrip()
            wt, ws = line.split()
            try:
                vec_t = target_model[wt]
                vec_s = source_model[ws]
            except:
                continue

            if matrix_t is None:
                matrix_t = vec_t
                matrix_s = vec_s
            else:
                matrix_t = np.vstack((matrix_t, vec_t))
                matrix_s = np.vstack((matrix_s, vec_s))

        matrix_s_T = matrix_s.T

        trans_matrix_1 = np.dot(matrix_s_T, matrix_s)
        trans_matrix_2 = np.dot(matrix_s_T, matrix_t)
        trans_matrix_1 = np.linalg.inv(trans_matrix_1)

        trans_matrix = np.dot(trans_matrix_1, trans_matrix_2)

        # x = np.dot(trans_matrix, vec_s)
        # print(vec_t[:10])
        # print(x[:10])
        # print(cosine_similarity(x[:5], vec_t[:5]))
        # print(cosine_similarity(x[:10], vec_t[:10]))
        # print(cosine_similarity(x, vec_t))
    return trans_matrix.T


def compute_transform_matrix_regression2(target_model, source_model, filename):
    with open(filename, encoding="utf-8") as f:
        matrix_t = None
        matrix_s = None
        for line in f:
            line = line.rstrip()
            wt, ws = line.split()
            try:
                vec_t = target_model[wt]
                vec_s = source_model[ws]
            except:
                continue

            if matrix_t is None:
                matrix_t = vec_t
                matrix_s = vec_s
            else:
                matrix_t = np.vstack((matrix_t, vec_t))
                matrix_s = np.vstack((matrix_s, vec_s))

        from scipy.linalg import lstsq

        # Přidání sloupce jedniček k původním vektorům v jazyce A pro zahrnutí prahu do transformace
        # matrix_t = np.hstack((matrix_t, np.ones((matrix_t.shape[0], 1))))

        # Vytvoření transformační matice pomocí metody nejmenších čtverců
        W, _, _, _ = lstsq(matrix_t, matrix_s)

        # Transformace vektoru z jazyka A do jazyka B
        # X_A_transformed = np.dot(vec_s, W)
        # print(cosine_similarity(X_A_transformed, vec_t))
        # X_A_transformed = np.dot(W, vec_s)
        # print(cosine_similarity(X_A_transformed, vec_t))
        return W

def compute_transform_matrix_orthogonal(target_model, source_model, filename):
    with open(filename, encoding="utf-8") as f:
        matrix_t = None
        matrix_s = None
        for line in f:
            line = line.rstrip()
            wt, ws = line.split()
            try:
                vec_t = target_model[wt]
                vec_s = source_model[ws]
            except:
                continue

            if matrix_t is None:
                matrix_t = vec_t
                matrix_s = vec_s
            else:
                matrix_t = np.vstack((matrix_t, vec_t))
                matrix_s = np.vstack((matrix_s, vec_s))

        matrix_t_T = matrix_t.T

        X = np.dot(matrix_t_T, matrix_s)
        U, s, V_T = np.linalg.svd(X)
        V = V_T.T[:, :len(s)]
        U_T = U.T
        trans_matrix = np.dot(V, U_T)
        # x = np.dot(trans_matrix.T, vec_s)
        # print(vec_t[:10])
        # print(x[:10])
        # print(cosine_similarity(x[:5], vec_t[:5]))
        # print(cosine_similarity(x[:10], vec_t[:10]))
        # print(cosine_similarity(x, vec_t))
    return trans_matrix.T

def compute_transform_matrix_orthogonal2(target_model, source_model, filename):
    with open(filename, encoding="utf-8") as f:
        matrix_t = None
        matrix_s = None
        for line in f:
            line = line.rstrip()
            wt, ws = line.split()
            try:
                vec_t = target_model[wt]
                vec_s = source_model[ws]
            except:
                continue

            if matrix_t is None:
                matrix_t = vec_t
                matrix_s = vec_s
            else:
                matrix_t = np.vstack((matrix_t, vec_t))
                matrix_s = np.vstack((matrix_s, vec_s))

        from scipy.linalg import orthogonal_procrustes

        trans_matrix, _ = orthogonal_procrustes(matrix_s, matrix_t)

    return trans_matrix.T

def compute_transform_matrix_procrustes_analysis(target_model, source_model, filename):
    with open(filename, encoding="utf-8") as f:
        matrix_t = None
        matrix_s = None
        for line in f:
            line = line.rstrip()
            wt, ws = line.split()
            try:
                vec_t = target_model[wt]
                vec_s = source_model[ws]
            except:
                continue

            if matrix_t is None:
                matrix_t = vec_t
                matrix_s = vec_s
            else:
                matrix_t = np.vstack((matrix_t, vec_t))
                matrix_s = np.vstack((matrix_s, vec_s))

        X_mean = np.mean(matrix_t, axis=0)
        Y_mean = np.mean(matrix_s, axis=0)
        X_centered = matrix_t - X_mean
        Y_centered = matrix_s - Y_mean

        # Singular value decomposition (SVD)
        U, _, VT = np.linalg.svd(Y_centered.T @ X_centered)

        # Výpočet transformační matice
        Z = U @ VT

        x = np.dot(Z, vec_s)
        print(cosine_similarity(x, vec_t))

        x = np.dot(Z.T, vec_s)
        print(cosine_similarity(x, vec_t))

    return Z.T




def get_trans_matrix(vec_model_train, vec_model_test, target_lang, source_lang, trans_method, filename):
    # filename = constants.DICT_FOLDER + f"{target_lang}-{source_lang}.0-5000m.txt"
    # print(filename)
    # if not os.path.exists(filename):
    #     util.exception(f"Dictionary {filename} not found.")
    #
    # print("orto1")
    # trans_matrix = compute_transform_matrix_orthogonal(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("orto2")
    # trans_matrix = compute_transform_matrix_orthogonal2(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("regr1")
    # trans_matrix = compute_transform_matrix_regression(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("regr2")
    # trans_matrix = compute_transform_matrix_regression2(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    #
    # filename = constants.DICT_FOLDER + f"{target_lang}-{source_lang}.5000-6500m.txt"
    # print(filename)
    # if not os.path.exists(filename):
    #     util.exception(f"Dictionary {filename} not found.")
    #
    # print("orto1")
    # trans_matrix = compute_transform_matrix_orthogonal(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("orto2")
    # trans_matrix = compute_transform_matrix_orthogonal2(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("regr1")
    # trans_matrix = compute_transform_matrix_regression(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("regr2")
    # trans_matrix = compute_transform_matrix_regression2(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    #
    #
    # filename = constants.DICT_FOLDER + f"{target_lang}-{source_lang}m.txt"
    # print(filename)
    # if not os.path.exists(filename):
    #     util.exception(f"Dictionary {filename} not found.")
    #
    # print("orto1")
    # trans_matrix = compute_transform_matrix_orthogonal(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("orto2")
    # trans_matrix = compute_transform_matrix_orthogonal2(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("regr1")
    # trans_matrix = compute_transform_matrix_regression(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("regr2")
    # trans_matrix = compute_transform_matrix_regression2(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    #
    #
    # filename = constants.DICT_FOLDER + f"{target_lang}-{source_lang}.txt"
    # print(filename)
    # if not os.path.exists(filename):
    #     util.exception(f"Dictionary {filename} not found.")
    #
    # print("orto1")
    # trans_matrix = compute_transform_matrix_orthogonal(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("orto2")
    # trans_matrix = compute_transform_matrix_orthogonal2(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("regr1")
    # trans_matrix = compute_transform_matrix_regression(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)
    # print("regr2")
    # trans_matrix = compute_transform_matrix_regression2(vec_model_train, vec_model_test, filename)
    # eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)

    if trans_method == "orto1":
        trans_matrix = compute_transform_matrix_orthogonal(vec_model_train, vec_model_test, filename)
    elif trans_method == "orto2":
        trans_matrix = compute_transform_matrix_orthogonal2(vec_model_train, vec_model_test, filename)
    elif trans_method == "regr1":
        trans_matrix = compute_transform_matrix_regression(vec_model_train, vec_model_test, filename)
    elif trans_method == "regr2":
        trans_matrix = compute_transform_matrix_regression2(vec_model_train, vec_model_test, filename)
    else:
        util.exception(f"Unknown transform matrix method {trans_method}")

    eval_similarity(vec_model_train, vec_model_test, target_lang, source_lang, trans_matrix)

    return trans_matrix
