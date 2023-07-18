import random


def select_subset(numbers, subset_size):
    if subset_size >= len(numbers):
        return numbers

    subset = random.sample(numbers, subset_size)
    return subset


numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
subset_size = 5

subset = select_subset(numbers, subset_size)
print(subset)
