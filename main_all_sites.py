# Copyright (c) 2017 Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.


from lib import models_siamese, graph, abide_utils
import numpy as np
import os
import time


def split_data(site, train_perc):
    """ Split data into training and test indices """
    train_indices = []
    test_indices = []

    for s in np.unique(site):
        # Make sure each site is represented in both training and test sets
        id_in_site = np.argwhere(site == s).flatten()

        num_nodes = len(id_in_site)
        train_num = int(train_perc * num_nodes)

        prng.shuffle(id_in_site)
        train_indices.extend(id_in_site[:train_num])
        test_indices.extend(id_in_site[train_num:])

    # print("Number of labeled samples %d" % len(train_indices))

    return train_indices, test_indices


def prepare_pairs(X, y, site, indices):
    """ Prepare the graph pairs before feeding them to the network """
    N, M, F = X.shape
    n_pairs = int(len(indices) * (len(indices) - 1) / 2)
    triu_pairs = np.triu_indices(len(indices), 1)

    X_pairs = np.ones((n_pairs, M, F, 2))
    X_pairs[:, :, :, 0] = X[indices][triu_pairs[0]]
    X_pairs[:, :, :, 1] = X[indices][triu_pairs[1]]

    site_pairs = np.ones(int(n_pairs))
    site_pairs[site[indices][triu_pairs[0]] != site[indices][triu_pairs[1]]] = 0

    y_pairs = np.ones(int(n_pairs))
    y_pairs[y[indices][triu_pairs[0]] != y[indices][triu_pairs[1]]] = 0  # -1

    print(n_pairs)

    return X_pairs, y_pairs, site_pairs


rs = 1234

print("Random state is %d" % rs)
prng = np.random.RandomState(rs)

# Get subject features
atlas = 'ho'
kind = 'correlation'

subject_IDs = abide_utils.get_ids()
# Get all subject networks
networks = abide_utils.load_all_networks(subject_IDs, kind, atlas_name=atlas)
X = np.array(networks)

# Number of nodes
nodes = X.shape[1]

# Get ROI coordinates
coords = abide_utils.get_atlas_coords(atlas_name=atlas)

# Get subject labels
label_dict = abide_utils.get_subject_label(subject_IDs, label_name='DX_GROUP')
y = np.array([int(label_dict[x])-1 for x in sorted(label_dict)])

# Get site ID
site = abide_utils.get_subject_label(subject_IDs, label_name='SITE_ID')
unq = np.unique(list(site.values())).tolist()
site = np.array([unq.index(site[x]) for x in sorted(site)])

# Choose site IDs to include in the analysis
site_mask = range(20)
X = X[np.in1d(site, site_mask)]
y = y[np.in1d(site, site_mask)]
site = site[np.in1d(site, site_mask)]

# Split into training, validation and testing sets
training_num = 720
tr_idx, test_idx = split_data(site, 0.8)

prng.shuffle(test_idx)
subs_to_add = training_num - len(tr_idx)  # subjects that need to be moved from testing to training set
tr_idx.extend(test_idx[:subs_to_add])
test_idx = test_idx[subs_to_add:]
print("The test indices are the following: ")
print(test_idx)

all_combs = []
tr_mat = np.array(tr_idx).reshape([int(training_num / 6), 6])
for i in range(3):
    x1 = tr_mat[:, i * 2].flatten()
    x2 = tr_mat[:, i * 2 + 1].flatten()
    combs = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])
    all_combs.append(combs)

all_combs = np.vstack(all_combs)

# print(all_combs.shape)
n, m, f = X.shape
X_train = np.ones((all_combs.shape[0], m, f, 2), dtype=np.float32)
y_train = np.ones(all_combs.shape[0], dtype=np.int32)
site_train = np.ones(all_combs.shape[0], dtype=np.int32)

for i in range(all_combs.shape[0]):
    X_train[i, :, :, 0] = X[all_combs[i, 0], :, :]
    X_train[i, :, :, 1] = X[all_combs[i, 1], :, :]
    if y[all_combs[i, 0]] != y[all_combs[i, 1]]:
        y_train[i] = 0  # -1
    if site[all_combs[i, 0]] != site[all_combs[i, 1]]:
        site_train[i] = 0

print("Training samples shape")
print(X_train.shape)


# Get the graph structure
dist, idx = graph.distance_scipy_spatial(coords, k=10, metric='euclidean')
A = graph.adjacency(dist, idx).astype(np.float32)

graphs = []
for i in range(3):
    graphs.append(A)

# Calculate Laplacians
L = [graph.laplacian(A, normalized=True) for A in graphs]

# Number of nodes in graph and features
print("Number of controls in the dataset: ")
print(y.sum())

# Prepare training testing and validation sets
X_test, y_test, site_test = prepare_pairs(X, y, site, test_idx)

n, m, f, _ = X_train.shape

# Graph Conv-net
features = 64
K = 3
params = dict()
params['num_epochs']     = 80
params['batch_size']     = 200
params['eval_frequency'] = X_train.shape[0] / (params['batch_size'] * 2)

# Building blocks.
params['filter']         = 'chebyshev5'
params['brelu']          = 'b2relu'
params['pool']           = 'apool1'

# Architecture.
params['F']              = [features, features]   # Number of graph convolutional filters.
params['K']              = [K, K]   # Polynomial orders.
params['p']              = [1, 1]     # Pooling sizes.
params['M']              = [1]    # Output dimensionality of fully connected layers.
params['input_features'] = f
params['lamda']          = 0.35
params['mu']             = 0.6

# Optimization.
params['regularization'] = 5e-3
params['dropout']        = 0.8
params['learning_rate']  = 1e-3
params['decay_rate']     = 0.95
params['momentum']       = 0
params['decay_steps']    = X_train.shape[0] / params['batch_size']

params['dir_name']       = 'siamese_' + time.strftime("%Y_%m_%d_%H_%M") + '_feat' + str(params['F'][0]) + '_' + \
                           str(params['F'][1]) + '_K' + str(K) + '_state' + str(rs)

# Save logs to folder
path = os.path.dirname(os.path.realpath(__file__))
log_path = os.path.join(path, 'logs', params['dir_name'])
os.makedirs(log_path)
print(params)

# Run model
model = models_siamese.siamese_cgcnn_cor(L, **params)
accuracy, loss, t_step, scores_summary = model.fit(X_train, y_train, site_train, X_test, y_test, site_test)
print('Time per step: {:.2f} ms'.format(t_step*1000))

# Save training
tr_res = model.evaluate(X_train, y_train, site_train)
np.save(os.path.join(log_path, "X_train.npy"), X[tr_idx])
np.savetxt(os.path.join(log_path, "id_combs_train.txt"), all_combs, fmt='%d')
np.savetxt(os.path.join(log_path, "labels_train.txt"), y[tr_idx], fmt='%d')
np.savetxt(os.path.join(log_path, "site_train.txt"), site[tr_idx], fmt='%d')
np.savetxt(os.path.join(log_path, "scores_train.txt"), np.array(tr_res[3]), fmt='%.4f')
np.savetxt(os.path.join(log_path, "y_train.txt"), y_train, fmt='%d')

# Evaluate test data
print("Test accuracy is:")
res = model.evaluate(X_test, y_test, site_test)
print(res[0])

# Save testing
np.save(os.path.join(log_path, "X_test.npy"), X[test_idx])
np.savetxt(os.path.join(log_path, "labels_test.txt"), y[test_idx], fmt='%d')
np.savetxt(os.path.join(log_path, "site_test.txt"), site[test_idx], fmt='%d')
np.savetxt(os.path.join(log_path, "scores_test.txt"), np.array(res[3]), fmt='%.4f')
np.savetxt(os.path.join(log_path, "y_test.txt"), y_test, fmt='%d')
