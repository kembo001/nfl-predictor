import numpy as np

print("Numpy version:", np.__version__)

my_list = [1, 2, 3, 4, 5]

print(my_list * 2)

my_array = np.array([1, 2, 3, 4, 5])

print(my_array * 2)

print(type(my_array), type(my_list))

############### Multidimensional Arrays ###############

my_new_array = np.array([[['A', 'B', 'C'],['D', 'E', 'F'], ['G', 'H', 'I']],
                         [['A', 'B', 'C'],['D', 'E', 'F'], ['G', 'H', 'I']],
                          [['A', 'B', 'C'],['D', 'E', 'F'], ['G', 'H', 'I']]])

word = my_new_array[1, 1, 2] + my_new_array[0, 2, 2] + my_new_array[0, 0, 1]

print(word)
print()
############### Slicing ###############

my_slicing_array = np.array([[1, 2, 3, 4, 5],
                             [6, 7, 8, 9, 10],
                             [11, 12, 13, 14, 15],
                             [16, 17, 18, 19, 20],])

# array[start:end:step]
print(my_slicing_array[0])
print()
print(my_slicing_array[:2])
print()
print(my_slicing_array[::-1])
print()
print(my_slicing_array[:, 1:3])

print()
############### Arithmetic ###############
array = np.array([1, 2, 3])

# Scalar Arithmetic

print(array * -1)
print(array - 1)
print(array / 2)

# Vecorized Math Functions

print(np.sqrt(array))
print(np.round(array))
print(np.pi)

radii = np.array([1, 2, 3])
print(np.pi * radii **2)

# Element-wise artithmetic

array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6])

print(array1 + array2)
print(array1 - array2)
print(array1 * array2)
print(array1 / array2)
print(array1 ** array2)
print()
# Comparison operators

scores = np.array([91, 55, 100, 73, 82, 64])
print(scores == 100)
print(scores >= 65)
scores[scores < 65] = 0
print(scores)
print()
############### Broadcasting ###############

# Broadcastig allows Numoy to perform operations on arrays
# With different shapes by virtually expanding dimensions
# So they match the larger array's shape.

# The dimensions have the same size.
# OR
# One of the dimensions has a size of 1.

array3 = np.array([[1, 2, 3, 4]])
array4 = np.array([[1],[2],[3],[4]])

print(array3.shape)
print(array4.shape)

print(array3 * array4)
print()
print()
############### Aggregate Functions ###############

# Aggregate Functions = summarize data and typically return a single value
array5 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

print(np.sum(array5))
print(np.mean(array5))
print(np.std(array5))
print(np.min(array5))

print()
# Sum of columns
print(np.sum(array5, axis=0))
print()
# Sum of rows
print(np.sum(array5, axis=1))
print()
print()
############### Filtering ###############

# Filtering = Refers to the process of selecting elements from an array that match a given condition
ages = np.array([[21, 17, 19, 20, 16, 30, 18, 65],
                 [39, 22, 15, 99, 18, 19, 20, 21]])

teenagers = ages[ages < 18]
adults = ages[(ages >= 18) & (ages < 65)]
seniors = ages[ages >= 65]
print(teenagers)
print(adults)
print(seniors)

# Use the where clause when you want to preserve the original shape
adults2 = np.where(ages >= 18, ages, 0)

print(adults2)
print()
print()
############### Random Numbers ###############

# Use seed if you want to use the same random numbers again
rng = np.random.default_rng(seed = 1)

print(rng.integers(low=1, high=100, size = 3))
print()
print(rng.integers(low=1, high=100, size = (3,2)))

print(np.random.uniform(low = -1, high = 1, size = (3, 2)))
