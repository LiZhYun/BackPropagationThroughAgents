import numpy as np

test1 = np.random.rand(2,3,1)
test2 = test1.reshape(-1, 1)
test3 = test2.reshape(-1,3,1)
print(test1, test2, test2)