
import numpy as np
path = r'D:\trainAOAdata6'
data = np.loadtxt(path+r'\AOA\aoamap'+str(0)+'-'+str(0)+'-'+str(4)+r'\b'+'.txt')[:,2:4]
data = data/[360,90]
for i in range(5):
    for j in range(5):
        temp = np.loadtxt(path+r'\AOA\aoamap'+str(i)+'-'+str(j)+'-'+str(4)+r'\b'+'.txt')[:,2:4]
        temp = temp/[360,90]
        data=np.concatenate((data,temp),axis=1)
# path = r'D:\trainAOAdata5-2'
# # for i in range(5):
# #     for j in range(5):
# #         temp = np.loadtxt(path+r'\AOA\aoamap'+str(i)+'-'+str(j)+'-'+str(4)+r'\b'+'.txt')[:,2:4]
# #         temp = temp/[360,90]
# #         data=np.concatenate((data,temp),axis=1)
data = data[:,2:]
# # print(data.shape)
# path = r'D:\trainAOAdata5-1'
label = np.loadtxt(path+r'\AOA\aoamap0-0-4\label.txt')
np.savetxt(path+r'\Lasttraindata.txt',data[:,:])
np.savetxt(path+r'\Lasttrainlabel.txt',label[:,:])

np.savetxt(path+r'\Transtraindata.txt',data[::100,:])
np.savetxt(path+r'\Transtrainlabel.txt',label[::100,:])