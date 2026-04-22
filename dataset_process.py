import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()


# Function to filter dataset based on label groups
def filter_by_labels(group_labels):
    # Define the labels you want to keep: group_labels
    def filter_fn(image, label):
        # Check if the label is in the provided group
        return tf.reduce_any(tf.equal(label, group_labels))
    return filter_fn


# Function to normalize and resize the dataset
def normalize_image(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to range [0, 1]
    return image, label


# Resize image data to specific image size, not necessary
#def resize_image(image, label):
#   image = tf.image.resize(image, [image_size0, image_size1])  # image_size initialization in the run.py
#   return image, label


# regularization of train dataset
def train_ds_norm(train_ds, batch_size, shuffle_size):
    train_ds = train_ds.map(normalize_image)
    train_ds = train_ds.shuffle(shuffle_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    train_ds = tfds.as_numpy(train_ds)
    return train_ds


# regularization of test dataset
def test_ds_norm(test_ds, batch_size):
    test_ds = test_ds.map(normalize_image)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    test_ds = tfds.as_numpy(test_ds)
    return test_ds


# dataset upload
def ds_upload(ds_dir, ds_type):
    (train_dataset, test_dataset), info = tfds.load(ds_type, split=['train', 'test'], as_supervised=True, data_dir=ds_dir, with_info=True)
    return train_dataset, test_dataset


# Generation of dataloader and split into several groups
def gen_ds_load(group_label, const_params, train_dataset, test_dataset):
    batch_size = const_params['batch_size']
    num_task = const_params['num_task']
    shuffle_size = const_params['shuffle_size']

    # train/test ds list generate
    train_ds_list, test_ds_list = [], []
    for i in range(num_task):
        # Apply filtering to the dataset based on train groups
        train_ds = train_dataset.filter(filter_by_labels(group_label[i]))
        train_ds = train_ds_norm(train_ds, batch_size, shuffle_size)
        train_ds_list.append(train_ds)
        # Apply filtering to the dataset based on test groups
        test_ds = test_dataset.filter(filter_by_labels(group_label[i]))
        test_ds = test_ds_norm(test_ds, batch_size)
        test_ds_list.append(test_ds)
    return train_ds_list, test_ds_list


# Specific dataloader for -gHg-model-based similarity calculation
def gen_ds_load_neg_ghg(group_label, const_params, train_dataset, test_dataset):
    ghg_batch_size = const_params['ghg_batch_size']
    num_task = const_params['num_task']
    shuffle_size = const_params['shuffle_size']

    # train/test ds list generate
    train_ds_list, test_ds_list = [], []
    for i in range(num_task):
        # Apply filtering to the dataset based on train groups
        train_ds = train_dataset.filter(filter_by_labels(group_label[i]))
        train_ds = train_ds_norm(train_ds, ghg_batch_size, shuffle_size)
        train_ds_list.append(train_ds)
        # Apply filtering to the dataset based on test groups
        test_ds = test_dataset.filter(filter_by_labels(group_label[i]))
        test_ds = test_ds_norm(test_ds, ghg_batch_size)
        test_ds_list.append(test_ds)
    return train_ds_list, test_ds_list

