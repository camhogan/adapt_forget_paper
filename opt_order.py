import jax
import jax.numpy as jnp
import itertools
import random
import numpy as np


# random permutation for num_task
def perm(key, num_task):
    # List of num_task numbers
    num_list = []
    for i in range(num_task):
        num_list.append(i)
    numbers = jnp.array(num_list)

    # Shuffle the array using jax.random
    shuffled_numbers = jax.random.permutation(key, numbers)

    # Output the shuffled numbers
    return shuffled_numbers


# all possible permutations with P=3
def all_perm_task3():
    perm_list = [[0, 1, 2], [0, 2, 1], [1, 2, 0], [1, 0, 2], [2, 0, 1], [2, 1, 0]]
    return jnp.array(perm_list)


# randomly permutation of a label list from [0, (num_class-1)], used to define a random label list
def random_perm_label(num_class):
    # class_list of [0, 1, ... , num_class-1], representing labels
    class_list = []
    for i in range(num_class):
        class_list.append(i)

    # Generate all possible permutations of [0, 1, ... , num_class-1]
    permutations = list(itertools.permutations(class_list))

    # Randomly choose one permutation
    random_permutation = random.choice(permutations)

    return random_permutation


# min/max hamiltonian path order
def ham_path_order(num_task, rho_matrix):
    # return two orders in reverse order in min ham and max ham, please check independently
    # List of num_tasker to be original order
    org_order = []
    for i in range(num_task):
        org_order.append(i)

    # Get all permutations
    permutations = list(itertools.permutations(org_order))
    # hamiltonian path for different permutations
    ham_list = []
    for permutation in permutations:
        ham = 0
        for j in range(num_task - 1):
            ham += 1 - rho_matrix[permutation[j]][permutation[j + 1]]  # hamiltonian distance sum of 1-similarity between nearby tasks
        ham_list.append(ham)

    # Pair each element with its index
    indexed_numbers = list(enumerate(ham_list))
    # Sort based on the values (second element of each tuple)
    sorted_indexed_numbers = sorted(indexed_numbers, key=lambda x: x[1])
    # Extract the sorted list and the corresponding indices
    sorted_numbers = [x[1] for x in sorted_indexed_numbers]
    sorted_indices = [x[0] for x in sorted_indexed_numbers]

    # two orders in reverse order in min_ham and max_ham
    perm_min_path_list = [permutations[sorted_indices[0]], permutations[sorted_indices[1]]]
    perm_max_path_list = [permutations[sorted_indices[len(permutations) - 1]],
                          permutations[sorted_indices[len(permutations) - 2]]]
    return perm_min_path_list, perm_max_path_list


# train accuracy for core-periphery(cp/min_ham_center) and periphery-core(pc/max_ham_center) opt_order model
def peri_core_order(num_task, rho_matrix):
    # hamiltonian distance between each task with other tasks
    center_ham_list = []
    for i in range(num_task):
        center_ham = 0
        for j in range(num_task):
            if np.abs(j-i) > 0.5:
                center_ham += 1-rho_matrix[i][j]
        center_ham_list.append(center_ham)

    # Pair each element with its index
    indexed_list = list(enumerate(center_ham_list))

    # core-periphery model: Sort the list based on the values in ascending order
    cp_sorted_list = sorted(indexed_list, key=lambda x: x[1])
    # Extract the permutation in core-periphery model
    cp_perm = [x[0] for x in cp_sorted_list]

    # periphery-core: Sort the list based on the values in descending order
    pc_sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)
    # Extract the sorted values and their original indices
    pc_perm = [x[0] for x in pc_sorted_list]

    return cp_perm, pc_perm


# max hamiltonian path order with periphery-core node
def max_ham_path_pc_order(num_task, rho_matrix):
    # List of num_tasker to be original order
    org_order = []
    for i in range(num_task):
        org_order.append(i)

    # Get all permutations
    permutations = list(itertools.permutations(org_order))
    # hamiltonian path for different permutations
    ham_list = []
    for permutation in permutations:
        ham = 0
        for j in range(num_task - 1):
            ham += 1 - rho_matrix[permutation[j]][permutation[j + 1]]  # hamiltonian distance sum of 1-similarity between nearby tasks
        ham_list.append(ham)

    # Pair each element with its index
    indexed_numbers = list(enumerate(ham_list))
    # Sort based on the values (second element of each tuple)
    sorted_indexed_numbers = sorted(indexed_numbers, key=lambda x: x[1])
    # Extract the sorted list and the corresponding indices
    sorted_indices = [x[0] for x in sorted_indexed_numbers]

    # two orders in reverse order in min_ham and max_ham
    perm_max_path_list = [permutations[sorted_indices[len(permutations) - 1]],
                          permutations[sorted_indices[len(permutations) - 2]]]

    # find the periphery-core node within two max hamiltonian paths
    for perm_max_path in perm_max_path_list:
        perm1, perm2 = perm_max_path[0], perm_max_path[1]
        perm_last1, perm_last2 = perm_max_path[num_task-1], perm_max_path[num_task-2]
        if (1 - rho_matrix[perm1][perm2]) > (1 - rho_matrix[perm_last1][perm_last2]):
            return perm_max_path
