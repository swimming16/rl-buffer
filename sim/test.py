import numpy as np
# print(np.zeros(6))
# print(np.random.randint(1,55))
# with open("/home/xie/ai-buffer/sim/network/network.log",'r') as f:
#     for line in f:
#         p=line.split()
#         print(len(p))
# x = np.arange(10).reshape((2,5))
# x[1,:3]=np.array([8,8,8])
# print(x)
x=np.arange(10)
x1=np.reshape(x,(1,2,5))
print(x1.shape)