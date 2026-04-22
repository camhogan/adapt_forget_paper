import jax
import numpy as np
from flax.training import train_state
import optax
import tensorflow as tf
import csv
import time
import os
import sys

# import function from specific path, where stores written modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, "../../"))  # path relative to parent_dir with modules
sys.path.insert(0, parent_dir)
from dataset_process import ds_upload, gen_ds_load, gen_ds_load_neg_ghg
from file_extract import obtain_label_perm_avg, obtain_label_forget_avg
from opt_order import all_perm_task3, random_perm_label, perm, ham_path_order, peri_core_order
from continual_model import train_model, contin_learn_forget
from network import nn_index
from similarity import sim_matrix_zero_shot, sim_matrix_neg_ghg, sim_mean

###############################################################################################################
"""
parameters for experiment
"""
params = {
    # parameters for model choose
    'ds_type': 'cifar10',  # dataset type
    'nn_type': 'cnn2',  # nn model type: 'cnn2', 'cnn5', 'nonlinear2', 'nonlinear5'
    'sim_type': 'zero_shot',  # similarity calculation model type: 'zero_shot', '-ghg'
    'ghg_batch_size': 256,  # batch size to calculate similarity if 'sim_type' = '-ghg'

    # parameters for training process
    'num_task': 5,  # number of tasks
    'num_output_classes': 2,  # num of output classes
    'num_all_classes': 10,  # num of total classes in dataset (ex. 100 for cifar100)
    'learning_rate': 0.001,  # learning rate
    'num_regular_epochs': 5,  # number of epochs per task during regular training
    'num_continue_epochs': 5,  # number of epochs per task during continue training
    'batch_size': 4,   # batch size
    'shuffle_size': 1000,  # shuffle size
    'image_size': [32, 32, 3],  # size of image data, [28, 28] for grayscale image, [32, 32, 3] for colored ones

    # parameters for experiment setting
    'path2avg': 'cifar10_cnn2_P5_C2_forget_avg_index1',  # path to csv file you keep the average forget performance
    'num_index': 1,  # job index to submit, related to classes split and labels split
    'ini_seed': 0,  # initialized seed for model, 0 by default
    'key_index': 1,  # key index for jax.random
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

# parameters initialization in model
rng, inp_rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)  # PRNGKey for train_state initialization
ini_sample_input = (params['batch_size'], params['image_size'][0], params['image_size'][1], params['image_size'][2])  # batch_size, image shape

###############################################################################################################
"""
dataset upload and csv file extract
"""
Start = time.time()  # time record begin, not necessary
# Load the CIFAR-10 dataset using tensorflow_datasets
data_dir = '/tmp/tfds'
train_ds, test_ds = ds_upload(data_dir, params['ds_type'])

# obtain data from calculated .csv file
(org_label_list_split, target_label_list_split, acc_forget_avg_list) = obtain_label_forget_avg(params['path2avg'], params['num_task'], params['num_output_classes'])

"""
optimal forget calculation, with hamiltonian or periphery-core optimal order model
"""
# initialization of lists
test_forget_max_hp_list, test_forget_min_hp_list = [], []  # forget performance for hamiltonian path opt_order model
sim_avg_list = []  # average of similarity list

# continual training to obtain forget performance
for i in range(len(org_label_list_split)):
    # labels obtained from perm_avg csv file
    group_labels = org_label_list_split[i]
    target_random_labels = target_label_list_split[i]
    print("group labels:", group_labels)
    print("target_random_labels:", target_random_labels)

    # ds_list generation as original ds_list based on picked group_labels
    train_ds_list_org, test_ds_list_org = gen_ds_load(group_labels, params, train_ds, test_ds)

    # obtain all the trained model states after each task, so could be used in similarity computation
    train_model_state_list = []
    for j in range(params['num_task']):
        # Initialization of model and optimizer, inp_rng... should be initialized outside this function
        model = nn_index(params['nn_type'])
        optimizer = optax.adam(learning_rate=params['learning_rate'])
        model_params = model.init(jax.random.PRNGKey(params['ini_seed']), jax.random.normal(inp_rng, ini_sample_input))
        model_state = train_state.TrainState.create(apply_fn=model.apply, params=model_params, tx=optimizer)

        # train model for specific task, it's not continual learning, used in similarity calculation
        trained_model_state, loss_history, accuracy_history = train_model(model_state, train_ds_list_org[j],
                                                                          params['num_regular_epochs'], group_labels[j],
                                                                          target_random_labels[j], params['num_output_classes'])
        train_model_state_list.append(trained_model_state)
        del model
        del optimizer
        del model_params
        del model_state
        del trained_model_state

    # similarity matrix for 'sim_type' model
    if params['sim_type'] == 'zero_shot':
        sim_matrix = sim_matrix_zero_shot(params['num_task'], train_model_state_list, test_ds_list_org, group_labels,
                                          target_random_labels, params['num_output_classes'])
    elif params['sim_type'] == '-ghg':
        train_ds_list_full, test_ds_list_full = gen_ds_load_neg_ghg(group_labels, params, train_ds, test_ds)
        sim_matrix = sim_matrix_neg_ghg(params, train_ds_list_full, train_model_state_list, group_labels, target_random_labels)
    else:
        print("similarity type error")

    sim_avg = sim_mean(params['num_task'], sim_matrix)
    sim_avg_list.append(sim_avg)
    print("similarity matrix:")
    print(sim_matrix)
    print("similarity average:", sim_avg)

    """forget with hamiltonian path order model"""
    # permutations of minimum hamiltonian path and maximum hamiltonian path (two orders in reverse order for each path)
    perm_min_hp_list, perm_max_hp_list = ham_path_order(params['num_task'], sim_matrix)
    # obtain the average forget for two reverse orders in min/max hamiltonian path
    # min ham path
    test_forget_min_hp_avg = 0
    for perm_idx in range(len(perm_min_hp_list)):
        perm_min_hp = perm_min_hp_list[perm_idx]
        print("min hp perm" + str(perm_idx), perm_min_hp)
        # continual training
        train_ds_list_min_hp, test_ds_list_min_hp, group_labels_min_hp, target_random_labels_min_hp = [], [], [], []
        for k in range(params['num_task']):
            train_ds_list_min_hp.append(train_ds_list_org[perm_min_hp[k]])
            test_ds_list_min_hp.append(test_ds_list_org[perm_min_hp[k]])
            group_labels_min_hp.append(group_labels[perm_min_hp[k]])
            target_random_labels_min_hp.append(target_random_labels[perm_min_hp[k]])
        acc_forget_min_hp = contin_learn_forget(params, train_ds_list_min_hp, test_ds_list_min_hp, group_labels_min_hp, target_random_labels_min_hp)
        # forget of min hamiltonian path order
        test_forget_min_hp_avg += acc_forget_min_hp / len(perm_min_hp_list)

    # max ham path
    test_forget_max_hp_avg = 0
    for perm_idx in range(len(perm_max_hp_list)):
        perm_max_hp = perm_max_hp_list[perm_idx]
        print("max hp perm" + str(perm_idx), perm_max_hp)
        # continual training
        train_ds_list_max_hp, test_ds_list_max_hp, group_labels_max_hp, target_random_labels_max_hp = [], [], [], []
        for k in range(params['num_task']):
            train_ds_list_max_hp.append(train_ds_list_org[perm_max_hp[k]])
            test_ds_list_max_hp.append(test_ds_list_org[perm_max_hp[k]])
            group_labels_max_hp.append(group_labels[perm_max_hp[k]])
            target_random_labels_max_hp.append(target_random_labels[perm_max_hp[k]])
        acc_forget_max_hp = contin_learn_forget(params, train_ds_list_max_hp, test_ds_list_max_hp, group_labels_max_hp,
                                                target_random_labels_max_hp)
        # forget of max hamiltonian path order
        test_forget_max_hp_avg += acc_forget_max_hp / len(perm_max_hp_list)

    # forget performance collect
    test_forget_min_hp_list.append(test_forget_min_hp_avg)
    test_forget_max_hp_list.append(test_forget_max_hp_avg)

"""data saved to csv file"""
num_pick_classes = params['num_task']*params['num_output_classes']

# test.csv record
test_opt_forget_matrix = np.zeros((len(org_label_list_split), num_pick_classes+4))
# labels, average forget, forget in min path model, forget in max path model, average of similarities
for i in range(len(org_label_list_split)):
    for j in range(params['num_task']):
        for k in range(params['num_output_classes']):
            test_opt_forget_matrix[i][j * params['num_output_classes'] + k] = org_label_list_split[i][j][k]
    # forget_avg, min_hp, max_hp, sim_1-ghg
    test_opt_forget_matrix[i][num_pick_classes] = acc_forget_avg_list[i]
    test_opt_forget_matrix[i][num_pick_classes + 1] = test_forget_min_hp_list[i]
    test_opt_forget_matrix[i][num_pick_classes + 2] = test_forget_max_hp_list[i]
    test_opt_forget_matrix[i][num_pick_classes + 3] = sim_avg_list[i]

test_file_name = 'test_' + params['ds_type'] + '_' + params['nn_type'] + '_' + params['sim_type'] + '_' + 'P' + str(params['num_task']) + '_C' + str(params['num_output_classes']) + '_opt_forget_index' + str(params['num_index'])
with open(test_file_name + '.csv', mode="w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(test_opt_forget_matrix)
    
