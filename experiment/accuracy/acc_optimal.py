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
from file_extract import obtain_label_perm_avg
from opt_order import all_perm_task3, random_perm_label, perm, ham_path_order, peri_core_order
from continual_model import contin_train, train_model
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
    'image_size': [32, 32, 3],  # size of image data, [28, 28, 1] for grayscale image, [32, 32, 3] for colored ones

    # parameters for experiment setting
    'path2avg': 'cifar10_cnn2_P5_C2_perm_avg_index1',  # path to csv file you keep the average accuracy performance
    'num_index': 1,  # job index to submit
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
(org_label_list_split, target_label_list_split, acc_train_avg_list, acc_train_min_list, acc_train_max_list,
 acc_test_avg_list, acc_test_min_list, acc_test_max_list) = obtain_label_perm_avg(params['path2avg'], params['num_task'], params['num_output_classes'])

"""
accuracy calculation for different permutations, with hamiltonian or periphery-core optimal order model
"""
# initialization of lists
train_acc_max_hp_list, train_acc_min_hp_list = [], []  # train accuracy for hamiltonian path opt_order model
train_acc_pc_list, train_acc_cp_list = [], []  # train accuracy for periphery-core(pc) and core-periphery(cp) opt_order model
test_acc_max_hp_list, test_acc_min_hp_list = [], []  # test accuracy for hamiltonian path opt_order model
test_acc_pc_list, test_acc_cp_list = [], []  # test accuracy for periphery-core(pc) and core-periphery(cp) opt_order model
sim_avg_list = []  # average of similarity list

# continual training to obtain accuracy
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
        # if you want to calculate transfer error in train dataset, replace test_ds_list_org with train_ds_list_org
        sim_matrix = sim_matrix_zero_shot(params['num_task'], train_model_state_list, test_ds_list_org, group_labels,
                                          target_random_labels, params['num_output_classes'])
    elif params['sim_type'] == '-ghg':
        # dataset for similarity calculation with larger batch size
        train_ds_list_full, test_ds_list_full = gen_ds_load_neg_ghg(group_labels, params, train_ds, test_ds)
        sim_matrix = sim_matrix_neg_ghg(params, train_ds_list_full, train_model_state_list,
                                        group_labels, target_random_labels)
    else:
        print("similarity type no found")

    sim_avg = sim_mean(params['num_task'], sim_matrix)
    sim_avg_list.append(sim_avg)
    print("similarity matrix:")
    print(sim_matrix)
    print("similarity average:", sim_avg)

    """accuracies from hamiltonian path order model"""
    # permutations of minimum hamiltonian path and maximum hamiltonian path (two orders in reverse order for each path)
    perm_min_hp_list, perm_max_hp_list = ham_path_order(params['num_task'], sim_matrix)
    # obtain the average accuracy for two reverse orders in min/max hamiltonian path
    # min ham path
    train_acc_min_hp_avg, test_acc_min_hp_avg = 0, 0
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
        train_multi_task_acc_history_list, acc_train_list, acc_test_list, train_acc_min_hp, test_acc_min_hp = contin_train(params, train_ds_list_min_hp, test_ds_list_min_hp, group_labels_min_hp, target_random_labels_min_hp)
        # acc of min hamiltonian path order
        train_acc_min_hp_avg += train_acc_min_hp / len(perm_min_hp_list)
        test_acc_min_hp_avg += test_acc_min_hp / len(perm_min_hp_list)

    # max ham path
    train_acc_max_hp_avg, test_acc_max_hp_avg = 0, 0
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
        train_multi_task_acc_history_list, acc_train_list, acc_test_list, train_acc_max_hp, test_acc_max_hp = contin_train(params, train_ds_list_max_hp, test_ds_list_max_hp, group_labels_max_hp, target_random_labels_max_hp)
        # acc of max hamiltonian path order
        train_acc_max_hp_avg += train_acc_max_hp / len(perm_max_hp_list)
        test_acc_max_hp_avg += test_acc_max_hp / len(perm_max_hp_list)

    # acc collect
    train_acc_min_hp_list.append(train_acc_min_hp_avg), train_acc_max_hp_list.append(train_acc_max_hp_avg)
    test_acc_min_hp_list.append(test_acc_min_hp_avg), test_acc_max_hp_list.append(test_acc_max_hp_avg)

    """accuracies from core-periphery(cp/min_ham_center) and periphery-core(pc/max_ham_center) opt_order model"""
    perm_cp, perm_pc = peri_core_order(params['num_task'], sim_matrix)
    # obtain the average accuracy for cp and pc model with similarity matrix
    # core-periphery (min_ham_center)
    print("core-periphery perm:", perm_cp)
    train_ds_list_cp, test_ds_list_cp, group_labels_cp, target_random_labels_cp = [], [], [], []
    for k in range(params['num_task']):
        train_ds_list_cp.append(train_ds_list_org[perm_cp[k]])
        test_ds_list_cp.append(test_ds_list_org[perm_cp[k]])
        group_labels_cp.append(group_labels[perm_cp[k]])
        target_random_labels_cp.append(target_random_labels[perm_cp[k]])
    train_multi_task_acc_history_list, acc_train_list, acc_test_list, train_acc_cp, test_acc_cp = contin_train(params, train_ds_list_cp, test_ds_list_cp, group_labels_cp, target_random_labels_cp)

    # periphery-core (max_ham_center)
    print("periphery-core perm:", perm_pc)
    train_ds_list_pc, test_ds_list_pc, group_labels_pc, target_random_labels_pc = [], [], [], []
    for k in range(params['num_task']):
        train_ds_list_pc.append(train_ds_list_org[perm_pc[k]])
        test_ds_list_pc.append(test_ds_list_org[perm_pc[k]])
        group_labels_pc.append(group_labels[perm_pc[k]])
        target_random_labels_pc.append(target_random_labels[perm_pc[k]])
    train_multi_task_acc_history_list, acc_train_list, acc_test_list, train_acc_pc, test_acc_pc = contin_train(params, train_ds_list_pc, test_ds_list_pc, group_labels_pc, target_random_labels_pc)

    # acc collect
    train_acc_cp_list.append(train_acc_cp), train_acc_pc_list.append(train_acc_pc)
    test_acc_cp_list.append(test_acc_cp), test_acc_pc_list.append(test_acc_pc)

"""data saved to csv file"""
num_pick_classes = params['num_task']*params['num_output_classes']
# train.csv record
train_opt_perm_matrix = np.zeros((len(org_label_list_split), num_pick_classes+8))
# labels, acc_avg, acc_min, acc_max, acc_min_ham, acc_max_ham, acc_cp, acc_pc
for i in range(len(org_label_list_split)):
    for j in range(params['num_task']):
        for k in range(params['num_output_classes']):
            train_opt_perm_matrix[i][j * params['num_output_classes'] + k] = org_label_list_split[i][j][k]
    # acc_avg, acc_min, acc_max from results in multi-permutations
    train_opt_perm_matrix[i][num_pick_classes], train_opt_perm_matrix[i][num_pick_classes+1], train_opt_perm_matrix[i][num_pick_classes+2] = acc_train_avg_list[i], acc_train_min_list[i], acc_train_max_list[i]
    # acc_cp(min_center), acc_pc
    train_opt_perm_matrix[i][num_pick_classes+3], train_opt_perm_matrix[i][num_pick_classes+4] = train_acc_cp_list[i], train_acc_pc_list[i]
    # acc_min_hp, acc_max_hp
    train_opt_perm_matrix[i][num_pick_classes+5], train_opt_perm_matrix[i][num_pick_classes+6] = train_acc_min_hp_list[i], train_acc_max_hp_list[i]
    train_opt_perm_matrix[i][num_pick_classes+7] = sim_avg_list[i]
# put into csv file
train_file_name = 'train_' + params['ds_type'] + '_' + params['nn_type'] + '_' + params['sim_type'] + '_' + 'P' + str(params['num_task']) + '_C' + str(params['num_output_classes']) + '_opt_order_index' + str(params['num_index'])
with open(train_file_name + '.csv', mode="w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(train_opt_perm_matrix)

# test.csv record
test_opt_perm_matrix = np.zeros((len(org_label_list_split), num_pick_classes+8))
# labels, acc_avg, acc_min, acc_max, acc_min_ham, acc_max_ham, acc_cp, acc_pc
for i in range(len(org_label_list_split)):
    for j in range(params['num_task']):
        for k in range(params['num_output_classes']):
            test_opt_perm_matrix[i][j * params['num_output_classes'] + k] = org_label_list_split[i][j][k]
    # acc_avg, acc_min, acc_max from results in multi-permutations
    test_opt_perm_matrix[i][num_pick_classes], test_opt_perm_matrix[i][num_pick_classes+1], test_opt_perm_matrix[i][num_pick_classes+2] = acc_test_avg_list[i], acc_test_min_list[i], acc_test_max_list[i]
    # acc_cp(min_center), acc_pc
    test_opt_perm_matrix[i][num_pick_classes+3], test_opt_perm_matrix[i][num_pick_classes+4] = test_acc_cp_list[i], test_acc_pc_list[i]
    # acc_min_hp, acc_max_hp
    test_opt_perm_matrix[i][num_pick_classes+5], test_opt_perm_matrix[i][num_pick_classes+6] = test_acc_min_hp_list[i], test_acc_max_hp_list[i]
    test_opt_perm_matrix[i][num_pick_classes+7] = sim_avg_list[i]
# put into csv file
test_file_name = 'test_' + params['ds_type'] + '_' + params['nn_type'] + '_' + params['sim_type'] + '_' + 'P' + str(params['num_task']) + '_C' + str(params['num_output_classes']) + '_opt_order_index' + str(params['num_index'])
with open(test_file_name + '.csv', mode="w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(test_opt_perm_matrix)
    
