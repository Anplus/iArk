import torch
import math
from src.core import ConstantTerm as C
from src.dataprocess import FileDataSet
import numpy
import os
import matplotlib.pyplot as plt
import torchvision
from scipy import stats
import CenterCamera

class SquareArray:
    la = 16.44
    n_row = 8
    n_antenna = n_row * n_row
    antenna_cor = torch.Tensor(n_antenna, 2)
    antenna_cor_r = torch.Tensor(n_antenna, 2)  # 极坐标
    center_y = 0
    center_x = 0


    def __init__(self,n_row,center_y,center_x):
        self.center_y = center_y
        self.center_x = center_x
        self.n_row = n_row
        self.n_antenna = n_row*n_row
        self.antenna_cor = torch.Tensor(self.n_antenna, 2)
        self.antenna_cor_r = torch.Tensor(self.n_antenna,2)
        for i in range(self.n_antenna):
            y = (i % self.n_row) * self.la - (self.n_row - 1)/2.0 * self.la
            x = (i // self.n_row) * self.la - (self.n_row - 1)/2.0 * self.la
            y = -y+center_y
            x = x+center_x
            self.antenna_cor[i, 0] = x
            self.antenna_cor[i, 1] = y
            self.antenna_cor_r[i, 0] = math.sqrt(( x-center_x) * ( x-center_x) + (y-center_y) * (y-center_y))
            self.antenna_cor_r[i, 1] = math.atan2(y-center_y, x-center_x)

    def theta_theory_XYZ(self, X, Y, Z):
        dx = self.antenna_cor[:,0].view(1,-1)-X
        dy = self.antenna_cor[:,1].view(1,-1)-Y
        d = torch.sqrt(dx*dx+dy*dy+(Z)*(Z))
        return -d/C.lamb*2*math.pi

    def theta_theory(self, al, be):  # 1XN
        theta_k = self.antenna_cor_r[:, 1].contiguous().view(self.n_antenna,1)
        r = self.antenna_cor_r[:, 0].contiguous().view(self.n_antenna,1)
        theta_t = 2 * math.pi / C.lamb *r*torch.cos(al - theta_k)*torch.cos(be)
        return theta_t

    def p0(self, al, be, theta_m):
        theta_t = self.theta_theory(al, be)
        k = theta_t.size(1)
        delta = theta_m - theta_t
        cosd = torch.cos(delta).sum(1)
        sind = torch.sin(delta).sum(1)
        p = torch.sqrt(cosd * cosd + sind * sind) / k
        return p
    def DAp0(self,al, be, theta_m):
        theta_t = self.theta_theory(al, be)
        k = theta_t.size(1)
        delta = theta_m - theta_t
        delta2 = delta - delta[:,0]
        #print(delta2[0,1,:])
        cosd = (torch.cos(delta2)*torch.Tensor(stats.norm(0,0.1*1.414).cdf(delta2))).sum(1)
        sind = (torch.sin(delta2)*torch.Tensor(stats.norm(0,0.1*1.414).cdf(delta2))).sum(1)
        p = torch.sqrt(cosd * cosd + sind * sind) / k
        return p


def cal_offset(refanttena,trainPath):
    center_x = -57.54 + (0 * 2 + 8 - 1) / 2.0 * 16.44
    center_y = 57.54 - (0 * 2 + 8 - 1) / 2.0 * 16.44
    test = SquareArray(8,center_y,center_x)

    trainDataset = FileDataSet.FileDataset(trainPath + r'\traindata.txt',
                                           trainPath + r'\trainlabel2.txt')
    inputs, labels = trainDataset[:]
    # plt.plot(numpy.mod(inputs.numpy()[:, 24] - inputs.numpy()[:, 8], numpy.pi * 2))
    # plt.show()
    d = test.theta_theory_XYZ(torch.Tensor(labels[:, 0] *100).view(-1, 1), torch.Tensor(labels[:, 1] *100).view(-1, 1),
                              torch.Tensor(labels[:, 2] *100).view(-1, 1))
    print(d.shape)
    print(inputs.shape)

    # plt.plot(inputs.numpy()[:, 9])
    # plt.figure()
    #
    # plt.plot(labels.cpu().numpy()[0::1, 0])
    # plt.plot(labels.cpu().numpy()[0::1, 1])
    # plt.plot(
    #     numpy.mod(inputs.numpy()[:, 1] - inputs.numpy()[:, 8], numpy.pi * 2))
    # plt.plot(
    #     numpy.mod(inputs.numpy()[:, 0] - inputs.numpy()[:, 8], numpy.pi * 2))
    # plt.plot(numpy.mod((d.numpy()[:, 0] - d.numpy()[:, 8]), numpy.pi * 2))
    # plt.show()

    offset = numpy.zeros(shape=(64,1))
    for i in range(64):
        offset[i,0] = numpy.median(numpy.mod((inputs.numpy()[:, i] -inputs.numpy()[:, refanttena] - (d.numpy()[:, i] - d.numpy()[:, refanttena])), numpy.pi * 2))
    # plt.plot(offset)
    # plt.show()
    numpy.savetxt('offsetfromstatic.txt', offset)

def test_sub_anttena(row,colomn,length,trainPath):
    center_x = -57.54+(colomn*2+length-1)/2.0*16.44
    center_y = 57.54-(row*2+length-1)/2.0*16.44
    subAntenna = SquareArray(length, center_y, center_x)
    print(subAntenna.antenna_cor)
    subAntennaIndex = numpy.zeros(shape=length*length)
    for i in range(length*length):
        subAntennaIndex[i] = (i//length+colomn)*8+ i%length+row
    offset = numpy.loadtxt(r'offsetfromstatic.txt')
    offset = torch.Tensor(offset).view(1,-1)
    trainDataset = FileDataSet.FileDataset(trainPath + r'\traindata.txt',
                                           trainPath + r'\trainlabel2.txt')
    inputs, labels = trainDataset[:]
    inputs = inputs[:,subAntennaIndex]
    offset = offset[:,subAntennaIndex]
    inputs = inputs-offset
    labels = labels*100

    inputs = inputs.view(-1,length*length,1)
    w = 360
    h = 90
    Al = torch.linspace(0, w - 1, w).view(1, w)/180.0*numpy.pi
    Al = torch.matmul(torch.ones(h, 1), Al).view(1,1,w * h)
    Be = torch.linspace(0, h - 1, h).view(h, 1)/180.0*numpy.pi
    Be = torch.matmul(Be, torch.ones(1, w), ).view(1,1,w * h)

    altruth = numpy.arctan2(labels[:,1].numpy()-center_y,labels[:,0].numpy()-center_x)/numpy.pi*180
    betruth = numpy.arcsin((labels[:,2].numpy()+2)/numpy.sqrt((labels[:,1].numpy()-center_y)*(labels[:,1].numpy()-center_y)+(labels[:,0].numpy()-center_x)*(labels[:,0].numpy()-center_x)+(labels[:,2].numpy()+2)*(labels[:,2].numpy()+2)))/numpy.pi*180
    index = 10
    print(betruth[index])
    print(altruth[index])


    b=list()
    #plt.imshow(p0[index,:].view(h,w))
    imagepath = trainPath+r'\AOA\aoamap'+str(row)+'-'+str(colomn)+'-'+str(length)
    folder = os.path.exists(imagepath)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(imagepath)

    for i in range(0,labels.shape[0],10):
        index =  i
        a = subAntenna.p0(Al, Be, inputs[i, :, :].view(1, length * length, 1)).view(h, w)
        torchvision.utils.save_image(a, imagepath + '\\' + str(int(i/10)) + '.jpg')
        a = a.numpy()
        b.append([altruth[index],betruth[index],numpy.unravel_index(a.argmax(), a.shape)[1],numpy.unravel_index(a.argmax(), a.shape)[0]])
    #plt.show()
    b=numpy.array(b)
    numpy.savetxt(imagepath+r'\b.txt',numpy.array(b))
    numpy.savetxt(imagepath+r'\label.txt',labels[::10,].numpy())
def test_sub_anttena_Pixal(row,colomn,length,trainPath):
    center_x = -57.54+(colomn*2+length-1)/2.0*16.44
    center_y = 57.54-(row*2+length-1)/2.0*16.44
    subAntenna = SquareArray(length, center_y, center_x)
    print(subAntenna.antenna_cor)
    subAntennaIndex = numpy.zeros(shape=length*length)
    for i in range(length*length):
        subAntennaIndex[i] = (i//length+colomn)*8+ i%length+row
    offset = numpy.loadtxt(r'offsetfromstatic.txt')
    offset = torch.Tensor(offset).view(1,-1)
    trainDataset = FileDataSet.FileDataset(trainPath + r'\traindata.txt',
                                           trainPath + r'\trainlabel2.txt')
    inputs, labels = trainDataset[:]
    inputs = inputs[:,subAntennaIndex]
    offset = offset[:,subAntennaIndex]
    inputs = inputs-offset
    labels = labels*100

    inputs = inputs.view(-1,length*length,1)
    w = 600
    h = 600
    Al = torch.linspace(0, w - 1, w).view(1, w)
    Al = torch.matmul(torch.ones(h, 1), Al).view(1,1,w * h)
    Be = torch.linspace(0, h - 1, h).view(h, 1)
    Be = torch.matmul(Be, torch.ones(1, w), ).view(1,1,w * h)

    camera = CenterCamera.Camera
    Al,Be = camera.getAlBefromPixal(camera,Al,Be)
    altruth = numpy.arctan2(labels[:,1].numpy()-center_y,labels[:,0].numpy()-center_x)/numpy.pi*180
    betruth = numpy.arcsin((labels[:,2].numpy()+2)/numpy.sqrt((labels[:,1].numpy()-center_y)*(labels[:,1].numpy()-center_y)+(labels[:,0].numpy()-center_x)*(labels[:,0].numpy()-center_x)+(labels[:,2].numpy()+2)*(labels[:,2].numpy()+2)))/numpy.pi*180
    index = 10
    print(betruth[index])
    print(altruth[index])
    xtruth,ytruth = camera.getPixalFromAlBe(camera,altruth,betruth)


    b=list()
    #plt.imshow(p0[index,:].view(h,w))
    imagepath = trainPath+r'\AOAPix\aoamap'+str(row)+'-'+str(colomn)+'-'+str(length)
    folder = os.path.exists(imagepath)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(imagepath)

    for i in range(0,labels.shape[0],10):
        index =  i
        a = subAntenna.p0(Al, Be, inputs[i, :, :].view(1, length * length, 1)).view(h, w)
        torchvision.utils.save_image(a, imagepath + '\\' + str(int(i/10)) + '.jpg')
        a = a.numpy()
        b.append([xtruth[index],ytruth[index],numpy.unravel_index(a.argmax(), a.shape)[1],numpy.unravel_index(a.argmax(), a.shape)[0]])
    #plt.show()
    b=numpy.array(b)
    numpy.savetxt(imagepath+r'\b.txt',numpy.array(b))
    numpy.savetxt(imagepath+r'\label.txt',labels[::10,].numpy())




if __name__ == '__main__':
    # cal_offset(0)
    # offset = numpy.loadtxt(r'offsetfromstatic.txt')
    # trainP = r'D:\Documents\OptiTrack\7-9-2'
    # path = r'D:\Documents\OptiTrack\7-11-1'
    # for L in range(1):
    #     for i in range(5-L):
    #         for j in range(5-L):
    #             cal_offset(8 * j + i, r'D:\Documents\OptiTrack\7-12-1')
    #             test_sub_anttena(i, j, 4+L, path)
    #
    # path = r'D:\Documents\OptiTrack\7-12-2'
    # for L in range(1):
    #     for i in range(5 - L):
    #         for j in range(5 - L):
    #             cal_offset(8 * j + i, r'D:\Documents\OptiTrack\7-12-1')
    #             test_sub_anttena(i, j, 4 + L, path)
    path=r'D:\Documents\OptiTrack\devices\bpsk'
    i=0
    j=0
    L=8
    cal_offset(8 * j + i, path)
    test_sub_anttena(i, j, L, path)
    i = 2
    j = 2
    L = 4
    cal_offset(8 * j + i, path)
    test_sub_anttena(i, j, L, path)
    i = 3
    j = 3
    L = 2
    cal_offset(8 * j + i, path)
    test_sub_anttena(i, j, L, path)

    #cal_offset(8 * j + i, r'D:\Documents\OptiTrack\7-12-1')
    # test_sub_anttena(i, j, 2, path)
    # test_sub_anttena(i, j, 3, path)
    # test_sub_anttena(i, j, 4, path)
    # test_sub_anttena(i, j, 5, path)
    # test_sub_anttena(i, j, 6, path)
    # test_sub_anttena(i, j, 7, path)
    # test_sub_anttena(i, j, 8, path)
    # path = r'D:\distance\\'
    # for L in range(2):
    #     cal_offset(0, r'D:\Documents\OptiTrack\7-12-1')
    #     test_sub_anttena(0, 0, 8, path+str(L+6))









