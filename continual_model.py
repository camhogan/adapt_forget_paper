import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
from network import nn_index


# Using optax as optimer
@jax.jit
def apply_model(state, images, labels):
    """Computes gradients, loss and accuracy for a single batch."""
    def loss_fn(params):
        """cross-entropy loss function"""
        logits_model = state.apply_fn(params, images)
        print(logits_model.shape[1])
        one_hot = nn.one_hot(labels, logits_model.shape[1])
        loss_model = jnp.mean(optax.softmax_cross_entropy(logits=logits_model, labels=one_hot))
        return loss_model, logits_model

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    """gradient update model"""
    return state.apply_gradients(grads=grads)


def batch_label_change(batch_label_i, num_classes, org_group_label, random_classes_label):
    """map batch label to specific label based on the random label
    train_group_label: original labels of classes label in dataset
    random_classes_label: specific labels assigned from 0 to num_classes in classification, it could be random assigned
    """
    for i in range(num_classes):
        if batch_label_i == org_group_label[i]:
            batch_label_i = random_classes_label[i]
    return batch_label_i


def train_epoch(state, data_loader, train_group_label, random_classes_label, num_group_classes):
    """Train for a single epoch.
    train_group_label: original labels of classes label in dataset
    random_classes_label: specific labels assigned from 0 to num_classes in classification, it could be random assigned
    """
    epoch_loss, epoch_accuracy = [], []

    for batch_images, batch_labels in data_loader:
        for j in range(len(batch_labels)):
            batch_labels[j] = batch_label_change(batch_labels[j], num_group_classes, train_group_label, random_classes_label)
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
        state = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)
    train_loss = jnp.mean(jnp.array(epoch_loss))
    train_accuracy = jnp.mean(jnp.array(epoch_accuracy))
    return state, train_loss, train_accuracy


def train_epoch_sub_batch(subset_idx, num_subsets, state, data_loader, train_group_label, random_classes_label, num_group_classes):
    """Train for subset of dataset in a single epoch.
    subset_idx: instead of training all the batches, only trained the sub_idx batches which is subset of total dataset
    train_group_label: original labels of classes label in dataset
    random_classes_label: specific labels assigned from 0 to num_classes in classification, it could be random assigned
    """
    epoch_loss, epoch_accuracy = [], []

    for batch_idx, (batch_images, batch_labels) in enumerate(data_loader):
        # Use modulo operation to select batches for the current subset
        if batch_idx % num_subsets != subset_idx:
            for j in range(len(batch_labels)):
                batch_labels[j] = batch_label_change(batch_labels[j], num_group_classes, train_group_label, random_classes_label)
            grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
            state = update_model(state, grads)
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
    train_loss = jnp.mean(jnp.array(epoch_loss))
    train_accuracy = jnp.mean(jnp.array(epoch_accuracy))
    return state, train_loss, train_accuracy


def train_model(state, train_data_loader, num_epochs, train_group_label, random_classes_label, num_group_classes):
    """Train for a loop with specific epochs."""
    loss_history, accuracy_history = [], []
    index = 0
    for epoch in range(num_epochs):
        index += 1
        state, train_loss, train_accuracy = train_epoch(state, train_data_loader, train_group_label, random_classes_label, num_group_classes)
        loss_history.append(train_loss)
        accuracy_history.append(train_accuracy)
        if index == num_epochs:
            print(f'epoch: {epoch:03d}, train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4f}')
    return state, loss_history, accuracy_history


def test_model(trained_model_state, test_ds, test_group_label, random_classes_label, num_classes):
    """Test for trained model."""
    test_loss = []
    test_accuracy = []
    for batch in test_ds:
        # Average test accuracy
        test_images, test_labels = batch
        for j in range(len(test_labels)):
            test_labels[j] = batch_label_change(test_labels[j], num_classes, test_group_label, random_classes_label)
        _, loss, accuracy = apply_model(trained_model_state, test_images, test_labels)
        test_loss.append(loss)
        test_accuracy.append(accuracy)
    return jnp.mean(jnp.array(test_loss)), jnp.mean(jnp.array(test_accuracy))


def test_model_inter_task(num_task, trained_model_state_list, test_dataset_list, test_group_label_list, random_classes_label_list, num_classes):
    """test across another test ds with given train model"""
    loss_matrix = jnp.zeros((num_task, num_task))
    acc_matrix = jnp.zeros((num_task, num_task))
    for train_index in range(num_task):
        for test_index in range(num_task):
            loss, accuracy = test_model(trained_model_state_list[train_index], test_dataset_list[test_index],
                                        test_group_label_list[test_index], random_classes_label_list[test_index], num_classes)
            loss_matrix = loss_matrix.at[train_index, test_index].set(loss)
            acc_matrix = acc_matrix.at[train_index, test_index].set(accuracy)
    return loss_matrix, acc_matrix


def contin_train(const_params, train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, random_labels_ordered):
    """continual training model"""
    # model initialization parameters, maybe not necessary
    rng, inp_rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)  # PRNGKey for train_state initialization
    ini_sample_input = (const_params['batch_size'], const_params['image_size'][0], const_params['image_size'][1],
                        const_params['image_size'][2])  # batch_size, image shape

    # Initialization of model and optimizer
    model = nn_index(const_params['nn_type'])
    optimizer = optax.adam(learning_rate=const_params['learning_rate'])
    # inp_rng... should be initialized outside this function
    model_params = model.init(jax.random.PRNGKey(const_params['ini_seed']), jax.random.normal(inp_rng, ini_sample_input))
    model_state = train_state.TrainState.create(apply_fn=model.apply, params=model_params, tx=optimizer)

    # training
    train_multi_task_acc_history_list = []  # accuracy history during continual learning of multi_task
    print("continual training acc history report for specific order:")
    for i in range(const_params['num_task']):
        model_state, loss_history, accuracy_history = train_model(model_state, train_ds_list_ordered[i],
                                                                  const_params['num_continue_epochs'],
                                                                  group_labels_ordered[i], random_labels_ordered[i],
                                                                  const_params['num_output_classes'])
        train_multi_task_acc_history_list.append(accuracy_history)

    # acc of the train dataset
    acc_train_list, loss_train_mean, acc_train_mean = [], 0, 0
    for i in range(const_params['num_task']):
        loss, acc = test_model(model_state, train_ds_list_ordered[i],
                               group_labels_ordered[i], random_labels_ordered[i], const_params['num_output_classes'])
        acc_train_list.append(acc)
        loss_train_mean += loss / const_params['num_task']
        acc_train_mean += acc / const_params['num_task']

    # acc of the test dataset
    acc_test_list, loss_test_mean, acc_test_mean = [], 0, 0
    for i in range(const_params['num_task']):
        loss, acc = test_model(model_state, test_ds_list_ordered[i],
                               group_labels_ordered[i], random_labels_ordered[i], const_params['num_output_classes'])
        acc_test_list.append(acc)
        loss_test_mean += loss / const_params['num_task']
        acc_test_mean += acc / const_params['num_task']

    return train_multi_task_acc_history_list, acc_train_list, acc_test_list, acc_train_mean, acc_test_mean


def contin_train_epoch_print(const_params, train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, random_labels_ordered):
    """continual training model, but after each epoch of training, apply it to test dataset
    This function is used to show the accuracy instantly after each epoch of training

    num_subsets: we divide the dataset into num_subsets batches
    """
    # model initialization parameters, maybe not necessary
    rng, inp_rng, init_rng = jax.random.split(jax.random.PRNGKey(const_params['ini_seed']), 3)  # PRNGKey for train_state initialization
    ini_sample_input = (const_params['batch_size'], const_params['image_size'][0], const_params['image_size'][1],
                        const_params['image_size'][2])  # batch_size, image shape

    # Initialization of model and optimizer
    model = nn_index(const_params['nn_type'])
    optimizer = optax.adam(learning_rate=const_params['learning_rate'])
    # inp_rng... should be initialized outside this function
    model_params = model.init(jax.random.PRNGKey(const_params['ini_seed']), jax.random.normal(inp_rng, ini_sample_input))
    model_state = train_state.TrainState.create(apply_fn=model.apply, params=model_params, tx=optimizer)

    # training
    multi_task_train_acc_history_list, multi_task_test_acc_history_list = [], [] # test accuracy history during continual learning of multi_task recorded by epoch
    for i in range(const_params['num_task']):
        multi_task_train_acc_history_list.append([])
        multi_task_test_acc_history_list.append([])

    # epoch 0 scratch test accuracy
    for i in range(const_params['num_task']):
        train_loss, train_acc = test_model(model_state, train_ds_list_ordered[i],
                                           group_labels_ordered[i], random_labels_ordered[i],
                                           const_params['num_output_classes'])
        test_loss, test_acc = test_model(model_state, test_ds_list_ordered[i],
                                         group_labels_ordered[i], random_labels_ordered[i],
                                         const_params['num_output_classes'])
        multi_task_train_acc_history_list[i].append(train_acc)
        multi_task_test_acc_history_list[i].append(test_acc)

    # epochs training with subsets
    for i in range(const_params['num_task']):
        for j in range(const_params['num_continue_epochs']):
            for subset_idx in range(const_params['num_subsets']):
                model_state, train_loss, train_accuracy = train_epoch_sub_batch(subset_idx, const_params['num_subsets'], model_state, train_ds_list_ordered[i],
                                                                                group_labels_ordered[i], random_labels_ordered[i], const_params['num_output_classes'])
                # for each subset-trained model state, use it for acc record
                for k in range(const_params['num_task']):
                    train_loss, train_acc = test_model(model_state, train_ds_list_ordered[k], group_labels_ordered[k],
                                                       random_labels_ordered[k], const_params['num_output_classes'])
                    test_loss, test_acc = test_model(model_state, test_ds_list_ordered[k], group_labels_ordered[k],
                                                     random_labels_ordered[k], const_params['num_output_classes'])
                    multi_task_train_acc_history_list[k].append(train_acc)
                    multi_task_test_acc_history_list[k].append(test_acc)

    return multi_task_train_acc_history_list, multi_task_test_acc_history_list


def contin_learn_forget(const_params, train_ds_list_ordered, test_ds_list_ordered, group_labels_ordered, random_labels_ordered):
    """ average forget in continual learning """
    # model initialization parameters, maybe not necessary
    rng, inp_rng, init_rng = jax.random.split(jax.random.PRNGKey(0), 3)  # PRNGKey for train_state initialization
    ini_sample_input = (const_params['batch_size'], const_params['image_size'][0], const_params['image_size'][1],
                        const_params['image_size'][2])  # batch_size, image shape

    # Initialization of model and optimizer
    model = nn_index(const_params['nn_type'])
    optimizer = optax.adam(learning_rate=const_params['learning_rate'])
    # inp_rng... should be initialized outside this function
    model_params = model.init(jax.random.PRNGKey(const_params['ini_seed']), jax.random.normal(inp_rng, ini_sample_input))
    model_state = train_state.TrainState.create(apply_fn=model.apply, params=model_params, tx=optimizer)

    # training
    train_multi_task_acc_history_list = []  # accuracy history during continual learning of multi_task
    model_state_list = []  # save each model states after each task training
    print("continual training acc history report for specific order:")
    for i in range(const_params['num_task']):
        model_state, loss_history, accuracy_history = train_model(model_state, train_ds_list_ordered[i],
                                                                  const_params['num_continue_epochs'],
                                                                  group_labels_ordered[i], random_labels_ordered[i],
                                                                  const_params['num_output_classes'])
        model_state_list.append(model_state)
        train_multi_task_acc_history_list.append(accuracy_history)

    # acc forget of the test dataset: acc_final - acc_i
    acc_test_forget_mean = 0
    for i in range(const_params['num_task']-1):
        loss, acc_final = test_model(model_state, test_ds_list_ordered[i],
                                     group_labels_ordered[i], random_labels_ordered[i],
                                     const_params['num_output_classes'])
        loss, acc_task_i = test_model(model_state_list[i], test_ds_list_ordered[i], group_labels_ordered[i],
                                      random_labels_ordered[i], const_params['num_output_classes'])
        acc_test_forget_mean += (acc_task_i-acc_final) / (const_params['num_task']-1)

    return acc_test_forget_mean
