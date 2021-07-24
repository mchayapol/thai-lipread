import numpy as np

# p0 = (946.0,742.0)
# p1 = (991.0,749.0)
p0 = (2,1)
p1 = (2,2)
p2 = (0,-1)
dp = (p1[0]-p0[0],p1[1]-p0[1])
v = [p0,p1,p2,dp]
v = np.array(v)
inv = np.degrees(np.arctan2(*v.T[::-1])) % 360.0
print(v)
print(inv)