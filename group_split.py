import jax
import jax.numpy as jnp


# pick specific classes from all classes in the dataset and rearrange these labels into group [[C] for P]
def random_pick_into_groups(key, num_pick_classes=14, num_total_classes=100, num_task=7, group_size=2):
    # Step 1: Randomly pick num_pick_classes unique integers from the range 0-99/num_total_classes
    # Generate a random permutation of numbers from 0 to 99
    permutation = jax.random.permutation(key, jnp.arange(num_total_classes))

    # Select the first 14 unique integers from the permutation
    numbers = permutation[:num_pick_classes]

    # Step 2: Shuffle the selected numbers (not strictly necessary here since they're already random)
    # Re-shuffle to add another layer of randomness if desired
    # key, subkey = jax.random.split(key)
    # numbers = jax.random.permutation(subkey, numbers)

    # Step 3: Divide the picked classes numbers into num_task groups, each containing group_size numbers
    groups = numbers.reshape((num_task, group_size))

    return groups
