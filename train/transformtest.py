from numpy import *
from math import sqrt


# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector
def findABCD(M): #M = 4*3
    A = B = C = D = 0
    dd = zeros(16)
    t = 0
    for i in range(4):
        summ = 0
        for j in range(4):
            dis = M[i,:]-M[j,:]
            dis = sqrt(sum(dis**2))
            summ = summ + dis
            print(dis)
            dd[t] = dis
            t = t+1
        if abs(summ - 0.305 - 0.43 - 0.70) < 0.1:
            A = i
        elif abs(summ - 0.305 - 0.307 - 0.63) < 0.1:
            B = i
        elif abs(summ - 0.305 - 0.327 - 0.43) < 0.1:
            C = i
        elif abs(summ - 0.63 - 0.327 - 0.70) < 0.1:
            D = i
    return A,B,C,D,sort(dd)
def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = mean(A, axis=0)
    centroid_B = mean(B, axis=0)

    # centre the points
    AA = A - tile(centroid_A, (N, 1))
    BB = B - tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = transpose(AA) * BB

    U, S, Vt = linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        print
        "Reflection detected"
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    print
    t

    return R, t


path = r'D:\Documents\OptiTrack\devices\passive'
M=loadtxt(path+r'\refpoints.txt')
A,B,C,D,dd = findABCD(M)
print(A,B,C,D,dd)
MM = M[(A,B,C,D),:]
yellow = dd[8]/2
d = yellow-dd[5]/2
tartM=array([2*yellow - d, -(yellow-d),0,2*yellow-d,yellow-d,0,d,yellow-d,0,-2*yellow+d,yellow-d,0]).reshape((4,3))
ret_R, ret_t = rigid_transform_3D(mat(MM), mat(tartM))
print(ret_R)
print(ret_t)
A=mat(loadtxt(path+r'\trainlabel.txt'))
print(A.shape)
A2 = (ret_R * A.T) + tile(ret_t, (1, A.shape[0]))
A2 = A2.T
savetxt(path+r'\trainlabel2.txt',A2)
print(yellow)

