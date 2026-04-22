import random
import jax.numpy as jnp
import jax
from jax import device_put
import gc
import optax
from flax import linen as nn

from continual_model import test_model, apply_model, batch_label_change

#######################################################################################################################
"""
zero-shot similarity model based on error transfer
"""


# average of similarities in upper similarity matrix
def sim_mean(num_task, similarity_matrix):
    sim_total, index = 0, 0
    for i in range(num_task-1):
        for j in range(i+1, num_task, 1):
            index += 1
            sim_total += similarity_matrix[i][j]
    return sim_total/index


# applying taskA trained_model_state to taskB dataset but with shuffled labels
def loss_org_model(trained_model_state, dataset, num_group_class):
    # initialization of lists
    loss_org, label_list = [], []

    # label list construction
    for i in range(num_group_class):
        label_list.append(i)

    for batch in dataset:
        # Average test accuracy and loss function
        images, labels = batch
        for j in range(len(labels)):
            labels[j] = random.choice(label_list)  # shuffled labels in test_labels
        _, loss, accuracy = apply_model(trained_model_state, images, labels)
        loss_org.append(loss)
    return jnp.mean(jnp.array(loss_org))


# apply trained model state to dataset of another task to generate cross-task loss function matrix
def test_model_inter_task_matrix(num_task, trained_model_state_list, ds_list, ds_group_labels, target_labels, num_group_class):
    loss_matrix = jnp.zeros((num_task, num_task))
    acc_matrix = jnp.zeros((num_task, num_task))
    for train_index in range(num_task):
        for test_index in range(num_task):
            # loss & accuracy apply trained model in task train_index to test dataset in test_index
            loss, accuracy = test_model(trained_model_state_list[train_index], ds_list[test_index],
                                        ds_group_labels[test_index], target_labels[test_index], num_group_class)
            loss_matrix = loss_matrix.at[train_index, test_index].set(float(loss))
            acc_matrix = acc_matrix.at[train_index, test_index].set(float(accuracy))
    return loss_matrix, acc_matrix


# loss matrix applying taskA trained_model_state to taskB test dataset but with shuffled labels
def loss_org_matrix(num_task, trained_model_state_list, ds_list, num_group_class):
    loss_org_m = jnp.zeros((num_task, num_task))
    for train_index in range(num_task):
        for test_index in range(num_task):
            loss = loss_org_model(trained_model_state_list[train_index], ds_list[test_index], num_group_class)
            loss_org_m = loss_org_m.at[train_index, test_index].set(float(loss))
    return loss_org_m


# similarity matrix based on the zero shot model, apply train model to dataset
def sim_matrix_zero_shot(num_task, trained_model_state_list, ds_list, group_labels, target_labels, num_output_class):
    # inter-task loss matrix
    loss_matrix_inter_task, acc_matrix = test_model_inter_task_matrix(num_task, trained_model_state_list, ds_list,
                                                                      group_labels, target_labels,
                                                                      num_output_class)
    # original loss matrix
    loss_org_m = loss_org_matrix(num_task, trained_model_state_list, ds_list, num_output_class)

    # zero-shot similarity matrix
    sim_matrix = jnp.zeros((num_task, num_task))
    for train_index in range(num_task):
        for test_index in range(num_task):
            sim = 1-0.5*(jnp.sqrt(loss_matrix_inter_task[train_index, test_index]/loss_org_m[train_index, test_index])+jnp.sqrt(loss_matrix_inter_task[test_index, train_index]/loss_org_m[test_index, train_index]))
            sim_matrix = sim_matrix.at[train_index, test_index].set(sim)
    return sim_matrix


# transfer error function
def error_transfer(num_task, trained_model_state_list, test_ds_list, test_group_labels, target_labels, num_output_class):
    # inter-task loss matrix
    loss_matrix_inter_task, acc_matrix = test_model_inter_task_matrix(num_task, trained_model_state_list, test_ds_list,
                                                                      test_group_labels, target_labels,
                                                                      num_output_class)
    # original loss matrix
    loss_org_m = loss_org_matrix(num_task, trained_model_state_list, test_ds_list, num_output_class)

    error_matrix = jnp.zeros((num_task, num_task))  # error transfer matrix
    error_shuffle_matrix = jnp.zeros((num_task, num_task))    # error transfer matrix divided by shuffling errors
    for train_index in range(num_task):
        for test_index in range(num_task):
            err = loss_matrix_inter_task[train_index, test_index]
            err_shuffle = loss_matrix_inter_task[train_index, test_index] / loss_org_m[train_index, test_index]
            error_matrix = error_matrix.at[train_index, test_index].set(err)
            error_shuffle_matrix = error_shuffle_matrix.at[train_index, test_index].set(err_shuffle)
    return error_matrix, error_shuffle_matrix


#######################################################################################################################
"""
Similarity model, similarity = -gHg, where g is gradient while H is Hessian matrix

The gHg calculation is based on average of gradient in batches:
g = sum(g_batches)/B , H = sum(H_batches)/B, B is the number of batches
"""


@jax.jit
def loss_fn(params, state, images, labels):
    # loss function given state, images, and labels
    logits = state.apply_fn(params, images)
    one_hot = nn.one_hot(labels, logits.shape[1])
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss


def avg_grads(state_params, state, data_loader, group_labels, target_labels, num_output_class):
    """
    Average of batch-gradients given a specific state for dataset in another task.
    group_labels: the original labels of classes in classification
    target_labels: randomly given target labels in training
    num_output_class: number of classes in one classification
    """
    accumulated_grads = None
    num_batches = 0
    for batch_images, batch_labels in data_loader:
        num_batches += 1
        for j in range(len(batch_labels)):
            # change labels to target labels choice in classification tasks
            batch_labels[j] = batch_label_change(batch_labels[j], num_output_class, group_labels, target_labels)
        batch_gradients = jax.grad(loss_fn, argnums=0)(state_params, state, batch_images, batch_labels)
        # Accumulate gradients (sum them)
        if accumulated_grads is None:
            accumulated_grads = batch_gradients
        else:
            accumulated_grads = jax.tree_util.tree_map(lambda x, y: x + y, accumulated_grads, batch_gradients)
        del batch_gradients
        gc.collect()
    # Average the gradients across all batches (optional, depending on how you want to handle them)
    averaged_gradients = jax.tree_util.tree_map(lambda x: x / num_batches, accumulated_grads)
    del accumulated_grads
    gc.collect()
    return averaged_gradients


@jax.jit
def hessian_vector_product_batch(state_params, state, images, labels, grad):
    # Computes hessian-grad product
    def grad_fn(params):
        return jax.grad(loss_fn)(params, state, images, labels)

    # Compute the Hessian-vector product using jvp (Jacobian-vector product)
    _, hvp = jax.jvp(grad_fn, (state_params,), (grad,))
    return hvp


def avg_hvp(model_params, state, data_loader, group_labels, target_labels, num_output_class, grad):
    """Average of hessian vector product given a specific state for dataset in another task ."""
    hvp_sum = None
    batch_count = 0
    for batch_images, batch_labels in data_loader:
        batch_count += 1
        for j in range(len(batch_labels)):
            batch_labels[j] = batch_label_change(batch_labels[j], num_output_class, group_labels, target_labels)
        hvp_batch = hessian_vector_product_batch(model_params, state, batch_images, batch_labels, grad)
        # Accumulate gradients (sum them)
        hvp_batch_cpu = device_put(hvp_batch, device=jax.devices("cpu")[0])  # Move to CPU
        if hvp_sum is None:
            hvp_sum = hvp_batch_cpu  # Initialize the sum with the first batch
        else:
            hvp_sum = jax.tree_util.tree_map(lambda x, y: x + y, hvp_sum, hvp_batch_cpu)
        del hvp_batch
    # Calculate the average Hessian by dividing each element in the PyTree by `batch_count`
    average_hvp_cpu = jax.tree_util.tree_map(lambda x: x / batch_count, hvp_sum)  # Average over batches
    return average_hvp_cpu


# flatten the gradient to 1D vector for production
def flatten_gradients(gradients):
    return jnp.concatenate([jnp.ravel(g) for g in jax.tree_util.tree_leaves(gradients)])


# similarity matrix based on the gradient-Hession-gradient model: -g*Hg
def sim_matrix_neg_ghg(const_params, train_ds_list, trained_state_list, train_groups_label_list, target_random_label_list):
    # number of tasks in continual learning
    num_task = const_params['num_task']

    # gHg determined similarity matrix
    sim_neg_ghg_m = jnp.zeros((num_task, num_task))
    g_norm_m, Hg_norm_m = jnp.zeros((num_task, num_task)), jnp.zeros((num_task, num_task))
    for mu in range(num_task):
        for v in range(num_task):
            # gradient with trained state of task-mu on dataset of task-v
            grad_state_mu_ds_v = avg_grads(trained_state_list[mu].params, trained_state_list[mu], train_ds_list[v],
                                           train_groups_label_list[v], target_random_label_list[v],
                                           const_params['num_output_classes'])
            grad_state_mu_ds_v_cpu = device_put(grad_state_mu_ds_v, device=jax.devices("cpu")[0])

            # Hg: H should be based on full dataset, with trained state of task-mu on dataset of task-mu
            hvp = avg_hvp(trained_state_list[mu].params, trained_state_list[mu], train_ds_list[mu],
                          train_groups_label_list[mu], target_random_label_list[mu],
                          const_params['num_output_classes'], grad_state_mu_ds_v)

            # ghg similarity with (task-mu, task-v)
            grad_mu_v_flat, hvp_flat = flatten_gradients(grad_state_mu_ds_v_cpu), flatten_gradients(hvp) # 1D vector flat
            grad_mu_v_norm, hvp_norm = jnp.linalg.norm(grad_mu_v_flat, ord=2), jnp.linalg.norm(hvp_flat, ord=2)
            ghg_norm = jnp.dot(grad_mu_v_flat, hvp_flat)
            sim_neg_ghg_m = sim_neg_ghg_m.at[mu, v].set(-ghg_norm)

    # average of (mu, v) and (v, mu) in similarity matrix
    sim_neg_ghg_avg_m = jnp.zeros((num_task, num_task))
    for mu in range(num_task):
        for v in range(num_task):
            sim_neg_ghg_avg_m = sim_neg_ghg_avg_m.at[mu, v].set((sim_neg_ghg_m[mu, v] + sim_neg_ghg_m[v, mu])/2)

    return sim_neg_ghg_avg_m
