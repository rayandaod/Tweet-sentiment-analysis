# import os
# import sys
# import numpy as np
# import pickle
# import scipy.sparse as sp
# from itertools import groupby
# BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_PATH+'/..')
# import src.paths as paths
# import src.params as params
#
#
# def ALS(train, test, num_features, lambda_user, lambda_item, stop_criterion):
#     """Alternating Least Squares (ALS) algorithm."""
#     # init parameters
#     change = 1
#     error_list = [0, 0]
#
#     # set seed
#     # np.random.seed(params.SEED)
#
#     # init ALS
#     user_features, item_features = init_MF(train, num_features)
#
#     # get the number of non-zero ratings for each user and item
#     nnz_items_per_user, nnz_users_per_item = train.getnnz(axis=0), train.getnnz(axis=1)
#
#     # group the indices by row or column index
#     nz_train, nz_item_user_indices, nz_user_item_indices = build_index_groups(train)
#
#     # run ALS
#     print("\nStart the ALS algorithm...")
#     while change > stop_criterion:
#         # update user feature & item feature
#         user_features = update_user_feature(
#             train, item_features, lambda_user,
#             nnz_items_per_user, nz_user_item_indices)
#         item_features = update_item_feature(
#             train, user_features, lambda_item,
#             nnz_users_per_item, nz_item_user_indices)
#
#         error = compute_error(train, user_features, item_features, nz_train)
#         print("RMSE on training set: {}.".format(error))
#         error_list.append(error)
#         change = np.fabs(error_list[-1] - error_list[-2])
#
#     # evaluate the test error
#     nnz_row, nnz_col = test.nonzero()
#     nnz_test = list(zip(nnz_row, nnz_col))
#     rmse = compute_error(test, user_features, item_features, nnz_test)
#     print("Test RMSE after running ALS: {v}.".format(v=rmse))
#
#
# def group_by(data, index):
#     """group list of list by a specific index."""
#     sorted_data = sorted(data, key=lambda x: x[index])
#     groupby_data = groupby(sorted_data, lambda x: x[index])
#     return groupby_data
#
#
# def build_index_groups(train):
#     """build groups for nnz rows and cols."""
#     nz_row, nz_col = train.nonzero()
#     nz_train = list(zip(nz_row, nz_col))
#
#     grouped_nz_train_byrow = group_by(nz_train, index=0)
#     nz_row_colindices = [(g, np.array([v[1] for v in value]))
#                          for g, value in grouped_nz_train_byrow]
#
#     grouped_nz_train_bycol = group_by(nz_train, index=1)
#     nz_col_rowindices = [(g, np.array([v[0] for v in value]))
#                          for g, value in grouped_nz_train_bycol]
#     return nz_train, nz_row_colindices, nz_col_rowindices
#
#
# def init_MF(train, num_features):
#     """init the parameter for matrix factorization."""
#
#     n, _ = train.get_shape()
#
#     W = np.random.rand(n, num_features)
#     Z = np.random.rand(n, num_features)
#
#     # start by item features.
#     item_nnz = train.getnnz(axis=1)
#     item_sum = train.sum(axis=1)
#
#     for ind in range(n):
#         Z[0, ind] = item_sum[ind, 0] / item_nnz[ind]
#     return W, Z
#
#
# def compute_error(data, user_features, item_features, nz):
#     """compute the loss (MSE) of the prediction of nonzero elements."""
#     mse = 0
#     for row, col in nz:
#         item_info = item_features[:, row]
#         user_info = user_features[:, col]
#         mse += (data[row, col] - user_info.T.dot(item_info)) ** 2
#     return np.sqrt(mse / len(nz))
#
#
# def update_user_feature(
#         train, item_features, lambda_user,
#         nnz_items_per_user, nz_user_itemindices):
#     """update user feature matrix."""
#     """the best lambda is assumed to be nnz_items_per_user[user] * lambda_user"""
#     num_user = nnz_items_per_user.shape[0]
#     num_feature = item_features.shape[0]
#     lambda_I = lambda_user * sp.eye(num_feature)
#     updated_user_features = np.zeros((num_feature, num_user))
#
#     for user, items in nz_user_itemindices:
#         # extract the columns corresponding to the prediction for given item
#         M = item_features[:, items]
#
#         # update column row of user features
#         V = M @ train[items, user]
#         A = M @ M.T + nnz_items_per_user[user] * lambda_I
#         X = np.linalg.solve(A, V)
#         updated_user_features[:, user] = np.copy(X.T)
#     return updated_user_features
#
#
# def update_item_feature(
#         train, user_features, lambda_item,
#         nnz_users_per_item, nz_item_userindices):
#     """update item feature matrix."""
#     """the best lambda is assumed to be nnz_items_per_item[item] * lambda_item"""
#     num_item = nnz_users_per_item.shape[0]
#     num_feature = user_features.shape[0]
#     lambda_I = lambda_item * sp.eye(num_feature)
#     updated_item_features = np.zeros((num_feature, num_item))
#
#     for item, users in nz_item_userindices:
#         # extract the columns corresponding to the prediction for given user
#         M = user_features[:, users]
#         V = M @ train[item, users].T
#         A = M @ M.T + nnz_users_per_item[item] * lambda_I
#         X = np.linalg.solve(A, V)
#         updated_item_features[:, item] = np.copy(X.T)
#     return updated_item_features
#
#
# if __name__ == '__main__':
#     print("Loading cooccurrence matrix")
#     with open(paths.COOC_PICKLE, 'rb') as f:
#         cooc = pickle.load(f)
#     print("{} nonzero entries".format(cooc.nnz))
#     ALS(train, test, params.K, params.LAMBDA_USER, params.LAMBDA_ITEM, params.STOP_CRITERION)
