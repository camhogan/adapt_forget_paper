from flax import linen as nn


# CNN structure for binary classification
class cnn_binary(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)
        x = nn.log_softmax(x)
        return x


# CNN structure for five-class classification
class cnn_five(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=5)(x)
        x = nn.log_softmax(x)
        return x


# nonlinear multilayer neuro network for binary classification
class nolinear_binary(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)
        return nn.log_softmax(x)  # Log-softmax for numerical stability


# nonlinear multilayer neuro network five-class classification
class nolinear_five(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=5)(x)
        return nn.log_softmax(x)  # Log-softmax for numerical stability


# return neuro network based on nn_label
def nn_index(nn_type):
    nn_type_input = 'True'
    if nn_type == 'cnn2':
        nn_model = cnn_binary()
    elif nn_type == 'cnn5':
        nn_model = cnn_five()
    elif nn_type == 'nonlinear2':
        nn_model = nolinear_binary()
    elif nn_type == 'nonlinear5':
        nn_model = nolinear_five()
    else:
        nn_type_input = 'False'
        print("nn type error")

    if nn_type_input == 'True':
        return nn_model
