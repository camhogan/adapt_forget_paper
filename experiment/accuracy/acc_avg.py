import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import csv
import time
import os
import sys
import random

# import function from specific path, where stores modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))  # path relative to parent_dir with modules
sys.path.insert(0, parent_dir)
from dataset_process import ds_upload, gen_ds_load
from opt_order import all_perm_task3, perm
from group_split import random_pick_into_groups
from continual_model import contin_train

###############################################################################################################
"""
parameters for experiment
"""
params = {
    # parameters for model choose
    'ds_type': 'cifar10',  # dataset type: 'fashion_mnist', 'cifar10', 'cifar100'
    'nn_type': 'cnn2',  # neurowork model type: 'cnn2', 'cnn5', 'nonlinear2', 'nonlinear5'
    'sim_type': 'zero_shot',  # similarity calculation model type: zero_short, -ghg

    # parameters for training process
    'num_task': 3,  # number of tasks
    'num_output_classes': 2,  # num of output classes
    'num_all_classes': 10,  # num of total classes in dataset (ex. 100 for cifar100)
    'task_shift_severe': True,  # False: similar task structure, True: dramatic task shifts
    'optimizer': 'adam',  # optimizer: 'sgd', 'sgd_momentum', 'adam', 'adamw', 'rmsprop'
    'optimizer_list': ['adam'],  # list of optimizers to compare in one run
    'learning_rate': 0.001,  # learning rate
    'sgd_momentum': 0.9,  # used by sgd_momentum
    'sgd_nesterov': False,  # used by sgd_momentum
    'adam_b1': 0.9,  # used by adam
    'adam_b2': 0.999,  # used by adam
    'adam_eps': 1e-8,  # used by adam
    'adamw_b1': 0.9,  # used by adamw
    'adamw_b2': 0.999,  # used by adamw
    'adamw_eps': 1e-8,  # used by adamw
    'adamw_weight_decay': 1e-4,  # used by adamw
    'rmsprop_decay': 0.9,  # used by rmsprop
    'rmsprop_eps': 1e-8,  # used by rmsprop
    'rmsprop_initial_scale': 0.0,  # used by rmsprop
    'rmsprop_centered': False,  # used by rmsprop
    'rmsprop_momentum': 0.0,  # used by rmsprop
    'rmsprop_nesterov': False,  # used by rmsprop
    'num_regular_epochs': 5,  # number of epochs per task during regular training
    'num_continue_epochs': 5,  # number of epochs per task during continue training
    'batch_size': 4,   # batch size
    'shuffle_size': 1000,  # shuffle size
    'image_size': [32, 32, 3],  # size of image data, [28, 28, 1] for grayscale image, [32, 32, 3] for colored ones

    # parameters for experiment setting
    'num_pick': 1,  # number of sampled task sets
    'num_perm': 1,  # number of multi-permutations for each sample point: 6, 30, 50 for P = 3, 5, 7
    'num_seeds': 10,  # number of random seeds to average over
    'num_index': 1,  # job index to submit, related to classes split and labels split
    'ini_seed': 0,  # initialized seed for model, set to constant
}

###############################################################################################################
"""
pre-allocation and initialization of parameters, default and no need to change in general case 
"""
# job performed on GPU by default
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU not found")
else:
    print('Found GPU at: {}'.format(device_name))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
# pre_set of gpu use
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'  # Limits JAX memory usage to 50%
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Disable pre_allocation

#parameters initialization in model
rng, inp_rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)  # PRNGKey for train_state initialization
ini_sample_input = (params['batch_size'], params['image_size'][0], params['image_size'][1], params['image_size'][2])  # batch_size, image shape

###############################################################################################################
"""
main function running
"""
Start = time.time()  # time record begin, not necessary
# global seeds for deterministic behavior across runs
random.seed(params['ini_seed'])
np.random.seed(params['ini_seed'])
tf.random.set_seed(params['ini_seed'])

# Load the CIFAR-10 dataset using tensorflow_datasets
data_dir = '/tmp/tfds_acc'
train_ds, test_ds = ds_upload(data_dir, params['ds_type'])

optimizer_list = params.get('optimizer_list', [params['optimizer']])

# Precompute fixed task splits, label mappings, and task orders so every optimizer sees the exact same data stream.
scenarios = []
for i in range(params['num_pick']):
    # Fixed curriculum mode for CIFAR-10 with 3 binary tasks.
    if params['ds_type'] == 'cifar10' and params['num_task'] == 3:
        if params.get('task_shift_severe', False):
            # Severe shift across tasks:
            # 1) cat vs dog, 2) frog vs truck, 3) airplane vs ship
            group_labels = jnp.array([[3, 5], [6, 9], [0, 8]])
        else:
            # Mild shift across tasks (same animal-vs-vehicle structure):
            # 1) horse vs truck, 2) deer vs automobile, 3) dog vs airplane
            group_labels = jnp.array([[7, 9], [4, 1], [5, 0]])
    else:
        key_pick = jax.random.PRNGKey(i + int(params['num_index'] * params['num_pick']))
        group_labels = random_pick_into_groups(
            key_pick,
            num_pick_classes=params['num_task'] * params['num_output_classes'],
            num_total_classes=params['num_all_classes'],
            num_task=params['num_task'],
            group_size=params['num_output_classes'],
        )

    target_random_labels = []
    for task_idx in range(params['num_task']):
        key_label = jax.random.PRNGKey(
            task_idx + i * params['num_task'] + params['num_index'] * params['num_pick'] * params['num_task']
        )
        label_perm = jax.random.permutation(key_label, jnp.arange(params['num_output_classes']))
        target_random_labels.append(tuple(np.array(label_perm).tolist()))

    if params['ds_type'] == 'cifar10' and params['num_task'] == 3:
        # Keep a fixed switch sequence for severe vs non-severe comparisons.
        orders = [np.array([0, 1, 2])]
    elif params['num_task'] == 3:
        orders = [np.array(order) for order in all_perm_task3()]
    else:
        orders = []
        for j in range(params['num_perm']):
            key_perm = jax.random.PRNGKey(
                j + int(i * params['num_perm']) + params['num_index'] * params['num_perm'] * params['num_pick']
            )
            orders.append(np.array(perm(key_perm, params['num_task'])))

    scenarios.append(
        {
            "group_labels": group_labels,
            "target_random_labels": target_random_labels,
            "orders": orders,
            "seed_base": int(params['num_index'] * 100000 + i * 1000),
        }
    )

summary_rows = [[
    "optimizer",
    "train_avg_mean",
    "train_min_mean",
    "train_max_mean",
    "test_avg_mean",
    "test_min_mean",
    "test_max_mean",
    "run_time_sec",
]]

for optimizer_name in optimizer_list:
    params['optimizer'] = optimizer_name
    opt_start = time.time()
    print("running optimizer:", optimizer_name)
    base_ini_seed = params['ini_seed']
    seed_list = [base_ini_seed + s for s in range(params.get('num_seeds', 1))]

    # list initialization: label list, train accuracy list and test accuracy list
    org_label_list_split, target_label_list_split = [], []  # original labels list, training target label list
    acc_train_avg_list, acc_train_min_list, acc_train_max_list = [], [], []
    acc_test_avg_list, acc_test_min_list, acc_test_max_list = [], [], []

    # accuracy calculation for num_pick sample points
    for i in range(params['num_pick']):
        group_labels = scenarios[i]["group_labels"]
        print("group_labels:", group_labels)

        # fixed target label mapping for fair optimizer comparison
        target_random_labels = scenarios[i]["target_random_labels"]
        seed_train_avg_list, seed_train_min_list, seed_train_max_list = [], [], []
        seed_test_avg_list, seed_test_min_list, seed_test_max_list = [], [], []

        for seed_idx, seed_value in enumerate(seed_list):
            params['ini_seed'] = seed_value
            random.seed(seed_value)
            np.random.seed(seed_value)
            tf.random.set_seed(seed_value)

            # ds list generation with deterministic shuffling seed for fair optimizer comparison
            train_ds_list_org, test_ds_list_org = gen_ds_load(
                group_labels,
                params,
                train_ds,
                test_ds,
                seed_base=scenarios[i]["seed_base"] + seed_idx * 10000,
                reshuffle_each_iteration=False,
            )

            # accuracy calculation for fixed permutation list to obtain acc_avg, min and max
            acc_train_avg, acc_test_avg, acc_train_perm_list, acc_test_perm_list = 0, 0, [], []
            for order in scenarios[i]["orders"]:
                # reorder of dataset, labels based on task order
                train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, target_random_labels_ordered = [], [], [], []
                for k in range(params['num_task']):
                    train_ds_list_ordered.append(train_ds_list_org[order[k]])
                    test_ds_list_ordered.append(test_ds_list_org[order[k]])
                    group_labels_ordered.append(group_labels[order[k]])
                    target_random_labels_ordered.append(target_random_labels[order[k]])

                # continual training, only acc_train_task_avg and acc_test_task_avg here for continual learn performance
                print("continual train task order:", order, "seed:", seed_value)
                train_multi_task_acc_history_list, acc_train_history, acc_test_history, acc_train_task_avg, acc_test_task_avg \
                    = contin_train(params, train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, target_random_labels_ordered)
                acc_train_perm_list.append(acc_train_task_avg)
                acc_test_perm_list.append(acc_test_task_avg)
                acc_train_avg += acc_train_task_avg / len(scenarios[i]["orders"])
                acc_test_avg += acc_test_task_avg / len(scenarios[i]["orders"])

            seed_train_avg_list.append(acc_train_avg)
            seed_train_min_list.append(jnp.min(jnp.array(acc_train_perm_list)))
            seed_train_max_list.append(jnp.max(jnp.array(acc_train_perm_list)))
            seed_test_avg_list.append(acc_test_avg)
            seed_test_min_list.append(jnp.min(jnp.array(acc_test_perm_list)))
            seed_test_max_list.append(jnp.max(jnp.array(acc_test_perm_list)))

        # train accuracy record
        acc_train_avg_list.append(float(np.mean(np.array(seed_train_avg_list))))
        acc_train_min_list.append(float(np.mean(np.array(seed_train_min_list))))
        acc_train_max_list.append(float(np.mean(np.array(seed_train_max_list))))
        # test accuracy record
        acc_test_avg_list.append(float(np.mean(np.array(seed_test_avg_list))))
        acc_test_min_list.append(float(np.mean(np.array(seed_test_min_list))))
        acc_test_max_list.append(float(np.mean(np.array(seed_test_max_list))))
        # label list record
        org_label_list_split.append(group_labels)
        target_label_list_split.append(target_random_labels)
    params['ini_seed'] = base_ini_seed

    # data save original classes label, random target label in training, train acc (avg, min, max), test acc (avg, min, max)
    perm_avg_matrix = np.zeros((params['num_pick'], params['num_task']*params['num_output_classes']*2+3+3))
    n_labels = params['num_task']*params['num_output_classes']*2  # number of labels in csv
    for i in range(params['num_pick']):
        # original classes label record
        for j in range(params['num_task']):
            for k in range(params['num_output_classes']):
                perm_avg_matrix[i][j*params['num_output_classes']+k] = org_label_list_split[i][j][k]

        # target randomly set label record
        for j in range(params['num_task']):
            for k in range(params['num_output_classes']):
                perm_avg_matrix[i][params['num_task']*params['num_output_classes']+j*params['num_output_classes']+k] = target_label_list_split[i][j][k]
        perm_avg_matrix[i][n_labels], perm_avg_matrix[i][n_labels+1], perm_avg_matrix[i][n_labels+2] = acc_train_avg_list[i], acc_train_min_list[i], acc_train_max_list[i]
        perm_avg_matrix[i][n_labels+3], perm_avg_matrix[i][n_labels+4], perm_avg_matrix[i][n_labels+5] = acc_test_avg_list[i], acc_test_min_list[i], acc_test_max_list[i]
    file_name = params['ds_type'] + '_' + params['nn_type'] + '_' + params['optimizer'] + '_' + 'P' + str(params['num_task']) + '_C' + str(params['num_output_classes']) + '_perm_avg_index' + str(params['num_index'])
    with open(file_name + '.csv', mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(perm_avg_matrix)

    opt_time = time.time() - opt_start
    summary_rows.append([
        optimizer_name,
        float(np.mean(np.array(acc_train_avg_list))),
        float(np.mean(np.array(acc_train_min_list))),
        float(np.mean(np.array(acc_train_max_list))),
        float(np.mean(np.array(acc_test_avg_list))),
        float(np.mean(np.array(acc_test_min_list))),
        float(np.mean(np.array(acc_test_max_list))),
        opt_time,
    ])

summary_name = params['ds_type'] + '_' + params['nn_type'] + '_P' + str(params['num_task']) + '_C' + str(params['num_output_classes']) + '_optimizer_compare_acc_index' + str(params['num_index']) + '.csv'
with open(summary_name, mode="w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(summary_rows)

End = time.time()
print("time cost:", End-Start)
