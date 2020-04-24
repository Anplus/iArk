import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.core import Parameters as pm
import torch.nn.functional as F

from src.dataprocess import FileDataSet
import AntennaArray_2eye as aa

row = pm.row
colomn = pm.colomn
length = pm.length
center_x = -57.54+(colomn*2+length-1)/2.0*16.44
center_y = 57.54-(row*2+length-1)/2.0*16.44
subAntenna = aa.SquareArray(length, center_y, center_x)
print(subAntenna.antenna_cor)
subAntennaIndex = np.zeros(shape=length*length)
for i in range(length * length):
    subAntennaIndex[i] = (i // length + colomn) * 8 + i % length + row
offset = np.loadtxt(r'../core/offsetfromstatic.txt')
offset = torch.Tensor(offset).view(1, -1)
offset = offset[:,subAntennaIndex]
w = 360
h = 90
Al = torch.linspace(0, w - 1, w).view(1, w) / 180.0 * np.pi
Al = torch.matmul(torch.ones(h, 1), Al).view(1, 1, w * h)
Be = torch.linspace(0, h - 1, h).view(h, 1) / 180.0 * np.pi
Be = torch.matmul(Be, torch.ones(1, w), ).view(1, 1, w * h)
TestMoade = pm.learnMode
if __name__ == '__main__':
    #import TrainTestbed
    model = torch.load('b.core')
    model.eval()
    testPath = r'D:\Documents\OptiTrack\7-9-1'
    testDataset = FileDataSet.FileDataset(testPath + r'\traindata.txt',
                                          testPath +r'\trainlabel2.txt')
    testloader = torch.utils.data.DataLoader(testDataset, batch_size=1,
                                              shuffle=False, num_workers=0)
    criterion = nn.MSELoss()
#    modelAE = torch.load('c.core')

    # randindex = torch.linspace(1,100,100)#np.random.randint(0, 80, size=[10])
    # for i in randindex:
    #     inputs, labels = testDataset[int(i)]
    #     labels=labels.view(1,1,2)
    #     one_hot = torch.zeros(1, pm.picWidth).scatter_(1, labels.data[:,:,0],1)
    #     inputs, labels = Variable(inputs), Variable(one_hot.view(-1, pm.picWidth))
    #     if torch.cuda.is_available():
    #         inputs = inputs.cuda()
    #         labels = labels.cuda()
    #     outputs = core(Variable(inputs ))
    #     scal = (torch.Tensor([pm.picWidth, pm.picHeight]).view(1, 2)).cuda()
    #
    #     #print(outputs.data * scal)
    #     print(outputs.data)
    #     print(labels)
    #     print('========')

    # inputs, labels = testDataset[:]
    # outputs=core(Variable(inputs.cuda()))
    # #scal = (torch.Tensor([pm.picWidth, pm.picHeight]).view(1, 2)).cuda()
    # #np.savetxt('a.txt', (outputs.data*scal).cpu().numpy(), fmt='%.6f')
    # np.savetxt('a.txt', (outputs.data).cpu().numpy(), fmt='%.6f')
    # for epoch in range(1):  # loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(testloader, 0):
    #         # get the inputs
    #         inputs, labels = data
    #         one_hot = torch.zeros(labels.size(0), pm.picWidth).scatter_(1, labels.data[:, :, 0], 1)
    #         inputs, labels = Variable(inputs), Variable(one_hot.view(-1, 1, pm.picWidth))
    #
    #         # wrap them in Variable
    #         #inputs, labels = Variable(inputs ), Variable(labels /torch.Tensor([pm.picWidth,pm.picHeight]).view(1,2))
    #         if torch.cuda.is_available():
    #             inputs = inputs.cuda()
    #             labels = labels.cuda()
    #
    #
    #         outputs = core(inputs)
    #         loss = criterion(outputs, labels)
    #
    #
    #         # print statistics
    #         running_loss += loss.data
    #         if i % 20 == 19:  # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.5f' %
    #                   (epoch + 1, i + 1, running_loss / 20))
    #             running_loss = 0.0
    #             # break
    #
    # print('Finished Training')
    if TestMoade == pm.LearningMode.Classification1LabelHeatMap:
        inputs, labels = testDataset[:]

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
       #outputs = core(Variable(modelAE.encoder(inputs)))
        outputs = model(Variable((inputs)))
        a = outputs.cpu().detach().numpy().transpose()
        # x = F.softmax(torch.Tensor(a[:pm.OutputShape[0],:]),0)
        # values, indices = torch.max(x, 0)
        # y = F.softmax(torch.Tensor(a[pm.OutputShape[0]:, :]), 0)
        # values2, indices2 = torch.max(y, 0)
        # index =torch.cat((indices.view(-1,1),indices2.view(-1,1)),1)
        # np.savetxt('a.txt',index.numpy().astype(int))
        # np.savetxt('b.txt', x.numpy())
        plt.figure()
        b = labels.cpu().numpy()[0::10, 0]
        plt.plot(b / 10)
        # plt.figure()
        # for i in range(100):
        #     plt.imshow(a[:, i * 10].reshape(48, 64))
        #     plt.show()
        #plt.imshow(a[:, 1].reshape(48, 64))
        plt.imshow(np.sum(a[:, 0::10].reshape(48, 64, -1), 0))
        temp = np.unravel_index(np.argmax(a[:, 0::10], 0), (48, 64))
        plt.plot(temp[1])
        plt.figure()
        b = labels.cpu().numpy()[0::10, 1]
        plt.plot(b / 10)
        # plt.figure()
        # for i in range(100):
        #     plt.imshow(a[:, i * 10].reshape(48, 64))
        #     plt.show()
        # plt.imshow(a[:, 1].reshape(48, 64))
        plt.imshow(np.sum(a[:, 0::10].reshape(48, 64, -1), 1))
        temp = np.unravel_index(np.argmax(a[:, 0::10], 0), (48, 64))
        plt.plot(temp[0])
        plt.figure()
        r1=labels.cpu().numpy()[0::10, 1]/10-temp[0]
        r2=labels.cpu().numpy()[0::10, 0]/10-temp[1]
        plt.hist(np.abs(r1),100)
        plt.figure()
        plt.hist(np.abs(r2),100)
        plt.figure()
        plt.hist(np.abs(np.sqrt(r1*r1+r2*r2)), 100)
        plt.figure()
        plt.plot(np.sqrt(r1*r1+r2*r2))
        print(np.mean(np.abs(r1)))
        print(np.mean(np.abs(r2)))

        temp = np.unravel_index(np.argmax(a[:,:], 0), (48, 64))
        temp2 = np.hstack((temp[1].reshape(-1,1)*10,temp[0].reshape(-1,1)*10))
        np.savetxt('a.txt', temp2)
        # for i, data in enumerate(testloader, 0):
        #     if i % 100 != 0:
        #         continue
        #     inputs, labels = data
        #     if torch.cuda.is_available():
        #         inputs = inputs.cuda()
        #         labels = labels.cuda()
        #     outputs = core(Variable(inputs))
        #     a = outputs.view(-1, 3072).cpu().detach().numpy()
        #     b = labels.cpu().numpy()
        #     print(b)
        #     plt.imshow(a[:, :].reshape(48, 64))
        #     plt.show()
    elif TestMoade == pm.LearningMode.Regression:
        inputs, labels = testDataset[1:40000:100]


        if pm.dataMode == pm.DataMode.SquareMode:
            inputs = inputs.view(-1, 1, 8, 8)
            outputs = model(Variable((inputs)))
            outputs = outputs.view(-1, 3)
        elif pm.dataMode == pm.DataMode.AoAMap:
            inputs = inputs[:, subAntennaIndex]
            inputs = inputs - offset
            labels = labels *100
            inputs = inputs.view(-1, length * length, 1)
            inputs = subAntenna.p0(Al, Be, inputs.view(-1, length*length, 1)).view(-1, h, w)
            altruth = np.arctan2(labels[:,  1].numpy() - center_y,
                                    labels[:,  0].numpy() - center_x) / np.pi * 180
            betruth = np.arcsin((labels[:, 2].numpy() + 2) / np.sqrt(
                (labels[:,  1].numpy() - center_y) * (labels[:,  1].numpy() - center_y) + (
                        labels[:, 0].numpy() - center_x) * (labels[:,  0].numpy() - center_x) + (
                        labels[:,  2].numpy() + 2) * (labels[:,  2].numpy() + 2))) / np.pi * 180
            temp = np.concatenate((np.asarray(altruth).reshape(-1,1), np.asarray(betruth).reshape(-1,1)), axis=1)
            temp = np.mod(temp, 360)
            labels = torch.Tensor(temp)
            #plt.imshow(inputs[0,:,:])
            #plt.show()

        else:
            outputs = model(Variable((inputs)))


        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        # plt.hist(labels.cpu().numpy()[:, 0],100)
        # plt.show()

        outputs = model(Variable((inputs)))
        b = labels.cpu().numpy()[:, 0]
        plt.plot(b)
        plt.plot(outputs.detach().cpu().numpy()[:,0]*360)
        r1 = b - outputs.detach().cpu().numpy()[:,0]*360
        plt.figure()
        b = labels.cpu().numpy()[:, 1]
        plt.plot(b)
        plt.plot((outputs.detach().cpu().numpy()[:, 1])*90)
        r2 = b - ((outputs.detach().cpu().numpy()[:, 1])*90)
        temp2 = np.hstack((outputs.detach().cpu().numpy()[:,0].reshape(-1, 1) * pm.picWidth, outputs.detach().cpu().numpy()[:,1].reshape(-1, 1) * pm.picHeight))
        np.savetxt('a.txt', temp2)
        plt.figure()
        plt.hist(np.abs(r1), 100)
        plt.figure()
        plt.hist(np.abs(r2), 100)
        print(np.median(np.abs(r1)))
        print(np.median(np.abs(r2)))

    elif TestMoade == pm.LearningMode.Classification2LabelsOneHot:
        inputs, labels = testDataset[:]
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model(Variable(inputs))
        a = outputs.cpu().detach().numpy().transpose()
        x = F.softmax(torch.Tensor(a[pm.OutputShape[0]:,:]),0)
        #values, indices = torch.max(x, 0)
        #y = F.softmax(torch.Tensor(a[pm.OutputShape[0]:, :]), 0)
        plt.imshow(x*x)
        b = labels.cpu().numpy()[:, 1]
        plt.plot(b)
        aa=np.argmax(x[:, 0::1], 0)
        plt.plot(aa.numpy())
        plt.show()




    #plt.figure()



    plt.show()



