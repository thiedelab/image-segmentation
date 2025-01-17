import numpy as np
import matplotlib.pyplot as plt
def prefix_sums(arr,k):
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(1, n + 1):
        prefix[i] = prefix[i - 1] + arr[i - 1]
        
    return prefix
print(prefix_sums([0,2,2,2,0],6))
# sequence_lengths = []
# num_iters = 10_000
# for i in range(num_iters):

#     sequence = []
#     while(True):

#         sequence.append(roll_dice())

#         if sequence[-1] == 5:
#             break
#         elif sequence[-1] == 6:
#             sequence_lengths.append(len(sequence))
#             break

# print(sum(sequence_lengths)/len(sequence_lengths))

