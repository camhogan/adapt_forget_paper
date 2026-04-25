import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import csv
import time
import random

# import function from specific path, where stores modules
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))  # path relative to parent_dir with modules
sys.path.insert(0, parent_dir)
from dataset_process import ds_upload, gen_ds_load
from opt_order import perm
from group_split import random_pick_into_groups
from continual_model import contin_learn_forget

###############################################################################################################
"""
parameters for experiment
"""
# Old CIFAR-10 parameter block (kept for reference, commented out).
# params = {
#     # parameters for model choose
#     'ds_type': 'cifar10',  # dataset type: 'fashion_mnist', 'cifar10', 'cifar100'
#     'nn_type': 'cnn2',  # neurowork model type: 'cnn2', 'cnn5', 'nonlinear2', 'nonlinear5'
#     'sim_type': 'zero_shot',  # similarity calculation model type
#
#     # parameters for training process
#     'num_task': 3,  # number of tasks
#     'num_output_classes': 2,  # num of output classes
#     'num_all_classes': 10,  # num of total classes in dataset (ex. 100 for cifar100)
#     'task_shift_severe': True,  # False: similar task structure, True: dramatic task shifts
#     'optimizer': 'adam',  # optimizer: 'sgd', 'sgd_momentum', 'adam', 'adamw', 'rmsprop'
#     'optimizer_list': ['sgd', 'sgd_momentum', 'adam', 'adamw', 'rmsprop'],  # list of optimizers to compare in one run
#     'learning_rate': 0.001,  # learning rate
#     'sgd_momentum': 0.9,  # used by sgd_momentum
#     'sgd_nesterov': False,  # used by sgd_momentum
#     'adam_b1': 0.9,  # used by adam
#     'adam_b2': 0.999,  # used by adam
#     'adam_eps': 1e-8,  # used by adam
#     'adamw_b1': 0.9,  # used by adamw
#     'adamw_b2': 0.999,  # used by adamw
#     'adamw_eps': 1e-8,  # used by adamw
#     'adamw_weight_decay': 1e-4,  # used by adamw
#     'rmsprop_decay': 0.9,  # used by rmsprop
#     'rmsprop_eps': 1e-8,  # used by rmsprop
#     'rmsprop_initial_scale': 0.0,  # used by rmsprop
#     'rmsprop_centered': False,  # used by rmsprop
#     'rmsprop_momentum': 0.0,  # used by rmsprop
#     'rmsprop_nesterov': False,  # used by rmsprop
#     'num_regular_epochs': 5,  # number of epochs per task during regular training
#     'num_continue_epochs': 5,  # number of epochs per task during continue training
#     'batch_size': 4,   # batch size
#     'shuffle_size': 1000,  # shuffle size
#     'image_size': [32, 32, 3],  # size of image data, [28, 28, 1] for grayscale image, [32, 32, 3] for colored ones
#
#     # parameters for experiment setting
#     'num_pick': 1,  # number of sampled task sets
#     'num_perm': 1,  # number of multi-permutations for each sample point: 6, 30, 50 for P = 3, 5, 7
#     'num_seeds': 10,  # number of random seeds to average over
#     'num_index': 1,  # job index to submit, related to classes split and labels split
#     'ini_seed': 0,  # initialized seed for model, set to constant
# }

cifar100_params = {
    # parameters for model choose
    'ds_type': 'cifar100',  # dataset type: 'fashion_mnist', 'cifar10', 'cifar100'
    'nn_type': 'cnn2',  # neurowork model type: 'cnn2', 'cnn5', 'nonlinear2', 'nonlinear5'
    'sim_type': 'zero_shot',  # similarity calculation model type

    # parameters for training process
    'num_task': 50,  # number of binary tasks (all 50 pairs from CIFAR-100)
    'num_output_classes': 2,  # num of output classes (binary tasks)
    'num_all_classes': 100,  # num of total classes in dataset
    'task_shift_severe': True,  # only used by hardcoded CIFAR-10 branch
    'optimizer': 'adam',  # optimizer: 'sgd', 'sgd_momentum', 'adam', 'adamw', 'rmsprop'
    'optimizer_list': ['sgd', 'sgd_momentum', 'adam', 'adamw', 'rmsprop'],  # list of optimizers to compare in one run
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
    'num_regular_epochs': 1,  # number of epochs per task during regular training
    'num_continue_epochs': 1,  # number of epochs per task during continue training
    'batch_size': 4,   # batch size
    'shuffle_size': 1000,  # shuffle size
    'image_size': [32, 32, 3],  # CIFAR-100 image shape

    # parameters for experiment setting
    'num_pick': 1,  # number of sampled task sets
    'num_perm': 1,  # number of multi-permutations for each sample point: 6, 30, 50 for P = 3, 5, 7
    'num_seeds': 10,  # number of random seeds to average over
    'num_index': 1,  # job index to submit, related to classes split and labels split
    'ini_seed': 0,  # initialized seed for model, set to constant
}

params = cifar100_params

# Fixed deterministic CIFAR-100 binary task pairs (all 50 pairs, fixed order).
fixed_cifar100_group_labels = jnp.arange(100).reshape((50, 2))
fixed_cifar100_task_order = np.arange(50)

# Optional override for quick smoke tests without editing code:
# NUM_TASK_OVERRIDE=3 python -u experiment/forget/forget_avg.py
num_task_override = os.getenv("NUM_TASK_OVERRIDE")
if num_task_override:
    params['num_task'] = int(num_task_override)

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

# Load dataset using tensorflow_datasets
# Use a dedicated TFDS cache for forget runs so accuracy/forget jobs can run concurrently.
data_dir = '/tmp/tfds_forget'
train_ds, test_ds = ds_upload(data_dir, params['ds_type'])

optimizer_list = params.get('optimizer_list', [params['optimizer']])

# Precompute fixed task splits, label mappings, and task orders so every optimizer sees the exact same data stream.
scenarios = []
for i in range(params['num_pick']):
    if params['ds_type'] == 'cifar100' and params['num_task'] <= 50:
        group_labels = fixed_cifar100_group_labels[:params['num_task']]
    # Fixed curriculum mode for CIFAR-10 with 3 binary tasks.
    elif params['ds_type'] == 'cifar10' and params['num_task'] == 3:
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
        label_perm = jax.random.permutation(key_label, np.arange(params['num_output_classes']))
        target_random_labels.append(tuple(np.array(label_perm).tolist()))

    if params['ds_type'] == 'cifar100' and params['num_task'] <= 50:
        orders = [fixed_cifar100_task_order[:params['num_task']]]
    elif params['ds_type'] == 'cifar10' and params['num_task'] == 3:
        # Keep a fixed switch sequence for severe vs non-severe comparisons.
        orders = [np.array([0, 1, 2])]
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

summary_rows = [["optimizer", "forget_mean", "run_time_sec"]]

for optimizer_name in optimizer_list:
    params['optimizer'] = optimizer_name
    opt_start = time.time()
    print("running optimizer:", optimizer_name)
    base_ini_seed = params['ini_seed']
    seed_list = [base_ini_seed + s for s in range(params.get('num_seeds', 1))]

    # list initialization: label list, accuracy forgetted list
    org_label_list_split, target_label_list_split = [], []  # original labels list, training target label list
    acc_forget_list = []
    n_label_cols = params['num_task'] * params['num_output_classes']
    seed_detail_header = (
        ['pick_index', 'seed']
        + [f'orig_label_{k}' for k in range(n_label_cols)]
        + [f'target_label_{k}' for k in range(n_label_cols)]
        + ['forget_avg']
    )
    seed_detail_rows = [seed_detail_header]

    # forget calculation for num_pick sample points
    for i in range(params['num_pick']):
        group_labels = scenarios[i]["group_labels"]
        print("group_labels:", group_labels)

        # fixed target label mapping for fair optimizer comparison
        target_random_labels = scenarios[i]["target_random_labels"]
        flat_org_labels = [int(v) for pair in np.array(group_labels).tolist() for v in pair]
        flat_target_labels = [int(v) for pair in target_random_labels for v in pair]
        seed_forget_avg_list = []
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

            # forget calculation for num_perm permutations to obtain average forget after continual learning
            acc_forget_avg = 0
            for order in scenarios[i]["orders"]:

                # reorder of dataset, labels based on task order
                train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, target_random_labels_ordered = [], [], [], []
                for k in range(params['num_task']):
                    train_ds_list_ordered.append(train_ds_list_org[order[k]])
                    test_ds_list_ordered.append(test_ds_list_org[order[k]])
                    group_labels_ordered.append(group_labels[order[k]])
                    target_random_labels_ordered.append(target_random_labels[order[k]])

                # continual training for accuracy forget
                print("continual train task order:", order, "seed:", seed_value)
                acc_forget = contin_learn_forget(params, train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, target_random_labels_ordered)
                acc_forget_avg += acc_forget / len(scenarios[i]["orders"])
            seed_forget_avg_list.append(acc_forget_avg)
            seed_detail_rows.append(
                [i, int(seed_value)]
                + flat_org_labels
                + flat_target_labels
                + [float(acc_forget_avg)]
            )

        # label list record
        org_label_list_split.append(group_labels)
        target_label_list_split.append(target_random_labels)
        acc_forget_list.append(float(np.mean(np.array(seed_forget_avg_list))))
    params['ini_seed'] = base_ini_seed

    # data save original classes label, random target label in training, average forget in tasks
    perm_forget_avg_matrix = np.zeros((params['num_pick'], params['num_task']*params['num_output_classes']*2+1))
    n_labels = params['num_task']*params['num_output_classes']*2  # number of labels in csv
    for i in range(params['num_pick']):
        # original classes label record
        for j in range(params['num_task']):
            for k in range(params['num_output_classes']):
                perm_forget_avg_matrix[i][j*params['num_output_classes']+k] = org_label_list_split[i][j][k]

        # target randomly set label record
        for j in range(params['num_task']):
            for k in range(params['num_output_classes']):
                perm_forget_avg_matrix[i][params['num_task']*params['num_output_classes']+j*params['num_output_classes']+k] = target_label_list_split[i][j][k]
        perm_forget_avg_matrix[i][n_labels] = acc_forget_list[i]
    file_name = params['ds_type'] + '_' + params['nn_type'] + '_' + params['optimizer'] + '_' + 'P' + str(params['num_task']) + '_C' + str(params['num_output_classes']) + '_forget_avg_index' + str(params['num_index'])
    with open(file_name + '.csv', mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(perm_forget_avg_matrix)
    # Additional seed-level breakdown CSV with average row at bottom.
    metric_start = 2 + 2 * n_label_cols
    metric_matrix = np.array([row[metric_start:] for row in seed_detail_rows[1:]], dtype=float)
    metric_means = list(np.mean(metric_matrix, axis=0))
    seed_detail_rows.append(['AVG', '-'] + [''] * (2 * n_label_cols) + metric_means)
    seed_file_name = (
        params['ds_type'] + '_' + params['nn_type'] + '_' + params['optimizer'] + '_' + 'P'
        + str(params['num_task']) + '_C' + str(params['num_output_classes']) + '_forget_avg_seed_detail_index'
        + str(params['num_index'])
    )
    with open(seed_file_name + '.csv', mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(seed_detail_rows)

    opt_time = time.time() - opt_start
    summary_rows.append([optimizer_name, float(np.mean(np.array(acc_forget_list))), opt_time])

summary_name = params['ds_type'] + '_' + params['nn_type'] + '_P' + str(params['num_task']) + '_C' + str(params['num_output_classes']) + '_optimizer_compare_forget_index' + str(params['num_index']) + '.csv'
with open(summary_name, mode="w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(summary_rows)

End = time.time()
print("time cost:", End-Start)
