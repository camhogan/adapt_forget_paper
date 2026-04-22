import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import csv
import time
import os
import sys

# import function from specific path, where stores modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))  # path relative to parent_dir with modules
sys.path.insert(0, parent_dir)
from dataset_process import ds_upload, gen_ds_load
from opt_order import all_perm_task3, random_perm_label, perm
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
    'num_task': 5,  # number of tasks
    'num_output_classes': 2,  # num of output classes
    'num_all_classes': 10,  # num of total classes in dataset (ex. 100 for cifar100)
    'learning_rate': 0.001,  # learning rate
    'num_regular_epochs': 5,  # number of epochs per task during regular training
    'num_continue_epochs': 5,  # number of epochs per task during continue training
    'batch_size': 4,   # batch size
    'shuffle_size': 1000,  # shuffle size
    'image_size': [32, 32, 3],  # size of image data, [28, 28, 1] for grayscale image, [32, 32, 3] for colored ones

    # parameters for experiment setting
    'num_pick': 10,  # number of ways to randomly pick and group num_task*num_class classes from num_all_class
    'num_perm': 30,  # number of multi-permutations for each sample point: 6, 30, 50 for P = 3, 5, 7
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
# Load the CIFAR-10 dataset using tensorflow_datasets
data_dir = '/tmp/tfds'
train_ds, test_ds = ds_upload(data_dir, params['ds_type'])

# list initialization: label list, train accuracy list and test accuracy list
org_label_list_split, target_label_list_split = [], []  # original labels list, training target label list
acc_train_avg_list, acc_train_min_list, acc_train_max_list = [], [], []
acc_test_avg_list, acc_test_min_list, acc_test_max_list = [], [], []

# accuracy calculation for num_pick sample points
for i in range(params['num_pick']):
    # randomly pick and group num_task*num_class classes from num_all_class
    key_pick = jax.random.PRNGKey(i + int(params['num_index'] * params['num_pick']))  # PRNGKey to avoid same pick
    group_labels = random_pick_into_groups(key_pick, num_pick_classes=params['num_task']*params['num_output_classes'],
                                           num_total_classes=params['num_all_classes'], num_task=params['num_task'],
                                           group_size=params['num_output_classes'])  # original label of class in ds
    print("group_labels:", group_labels)

    # randomly perms of target labels in classification
    target_random_labels = [random_perm_label(params['num_output_classes']) for _ in group_labels]

    # ds_list generation as original ds_list based on picked group_labels
    train_ds_list_org, test_ds_list_org = gen_ds_load(group_labels, params, train_ds, test_ds)

    # accuracy calculation for num_perm permutations to obtain acc_avg, min and max
    acc_train_avg, acc_test_avg, acc_train_perm_list, acc_test_perm_list = 0, 0, [], []
    if params['num_task'] == 3:
        # for P=3, only six permutations, so it's special case
        perms_task3 = all_perm_task3()
        for j in range(len(perms_task3)):
            order = perms_task3[j]  # task order

            # reorder of dataset, labels based on task order
            train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, target_random_labels_ordered = [], [], [], []
            for k in range(params['num_task']):
                train_ds_list_ordered.append(train_ds_list_org[order[k]])
                test_ds_list_ordered.append(test_ds_list_org[order[k]])
                group_labels_ordered.append(group_labels[order[k]])
                target_random_labels_ordered.append(target_random_labels[order[k]])

            # continual training, only acc_train_task_avg and acc_test_task_avg here for continual learn performance
            print("continual train task order:", order)
            train_multi_task_acc_history_list, acc_train_history, acc_test_history, acc_train_task_avg, acc_test_task_avg \
                = contin_train(params, train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered,
                               target_random_labels_ordered)
            acc_train_perm_list.append(acc_train_task_avg)
            acc_test_perm_list.append(acc_test_task_avg)
            acc_train_avg += acc_train_task_avg / len(perms_task3)
            acc_test_avg += acc_test_task_avg / len(perms_task3)
    else:
        for j in range(params['num_perm']):
            key_perm = jax.random.PRNGKey(j + int(i * params['num_perm']) + params['num_index']*params['num_perm']*params['num_pick'])  # PRNGKey of permutation
            order = perm(key_perm, params['num_task'])  # task order

            # reorder of dataset, labels based on task order
            train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, target_random_labels_ordered = [], [], [], []
            for k in range(params['num_task']):
                train_ds_list_ordered.append(train_ds_list_org[order[k]])
                test_ds_list_ordered.append(test_ds_list_org[order[k]])
                group_labels_ordered.append(group_labels[order[k]])
                target_random_labels_ordered.append(target_random_labels[order[k]])

            # continual training, only acc_train_task_avg and acc_test_task_avg here for continual learn performance
            print("continual train task order:", order)
            train_multi_task_acc_history_list, acc_train_history, acc_test_history, acc_train_task_avg, acc_test_task_avg \
                = contin_train(params, train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, target_random_labels_ordered)
            acc_train_perm_list.append(acc_train_task_avg)
            acc_test_perm_list.append(acc_test_task_avg)
            acc_train_avg += acc_train_task_avg / params['num_perm']
            acc_test_avg += acc_test_task_avg / params['num_perm']

    # train accuracy record
    acc_train_avg_list.append(acc_train_avg)
    acc_train_min_list.append(jnp.min(jnp.array(acc_train_perm_list)))
    acc_train_max_list.append(jnp.max(jnp.array(acc_train_perm_list)))
    # test accuracy record
    acc_test_avg_list.append(acc_test_avg)
    acc_test_min_list.append(jnp.min(jnp.array(acc_test_perm_list)))
    acc_test_max_list.append(jnp.max(jnp.array(acc_test_perm_list)))
    # label list record
    org_label_list_split.append(group_labels)
    target_label_list_split.append(target_random_labels)

    del train_ds_list_org
    del test_ds_list_org

# data save original classes label, random target label in training, train acc (avg, min, max), test acc (avg, min, max)
perm_avg_Matrix = np.zeros((params['num_pick'], params['num_task']*params['num_output_classes']*2+3+3))
n_labels = params['num_task']*params['num_output_classes']*2  # number of labels in csv
for i in range(params['num_pick']):
    # original classes label record
    for j in range(params['num_task']):
        for k in range(params['num_output_classes']):
            perm_avg_Matrix[i][j*params['num_output_classes']+k] = org_label_list_split[i][j][k]

    # target randomly set label record
    for j in range(params['num_task']):
        for k in range(params['num_output_classes']):
            perm_avg_Matrix[i][params['num_task']*params['num_output_classes']+j*params['num_output_classes']+k] = target_label_list_split[i][j][k]
    perm_avg_Matrix[i][n_labels], perm_avg_Matrix[i][n_labels+1], perm_avg_Matrix[i][n_labels+2] = acc_train_avg_list[i], acc_train_min_list[i], acc_train_max_list[i]
    perm_avg_Matrix[i][n_labels+3], perm_avg_Matrix[i][n_labels+4], perm_avg_Matrix[i][n_labels+5] = acc_test_avg_list[i], acc_test_min_list[i], acc_test_max_list[i]
file_name = params['ds_type'] + '_' + params['nn_type'] + '_' + 'P' + str(params['num_task']) + '_C' + str(params['num_output_classes']) + '_perm_avg_index' + str(params['num_index'])
with open(file_name + '.csv', mode="w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(perm_avg_Matrix)
End = time.time()
print("time cost:", End-Start)
