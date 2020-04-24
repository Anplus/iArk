import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time
import math
# import cupy as np
from scipy import signal
import pickle

from numba import jit
import os
from numba import vectorize

sampleRate = 6000000


@jit
def Add(a, b):
    return a + 1j * b


start = time.clock()
# x=B[2::4]+1j*B[3::4]


RN16_length_min = 7000


def CalTagPhase(data):
    data_abs = np.abs((data))
    diff_data = np.diff(data_abs)
    diff_abs = np.abs(np.diff(data_abs))
    diff_mean = np.mean(diff_abs)
    diff_max = np.max(diff_abs)
    # plt.figure()
    # plt.plot(np.abs(data))
    # plt.show()
    edge_index = np.where((diff_abs > diff_max * 0.8) & (diff_abs > 5 * diff_mean))[0]
    tagphase = []
    if (len(edge_index) == 0 or len(edge_index) > 2):
        return tagphase
    for i in edge_index:
        diff_edge = diff_data[i]
        IQedge = data[i + 1] - data[i]
        if (diff_edge < 0):
            IQedge = -IQedge
        tagphase.append(np.angle(IQedge))
    return tagphase


def IsEdge(x, ind, errT, threhold):
    for i in range(0, errT):
        if x[ind + i] > threhold:
            return i
    for i in range(-errT, 0):
        if x[ind + i] > threhold:
            return i
    return -100


def IsStart(x, ind, errT, threhold):
    if (ind > len(x) - 2000):
        return []
    result = []
    offset = IsEdge(x, ind, errT, threhold)
    if (offset != -100):
        result.append(ind + offset)
    ind = ind + 150 + offset
    offset = IsEdge(x, ind, errT, threhold)
    if (offset != -100):
        result.append(ind + offset)
        ind = ind + offset + 75
        offset = IsEdge(x, ind, errT, threhold)
        if (offset != -100):
            result.append(ind + offset)
            ind = ind + offset + 75
            offset = IsEdge(x, ind, errT, threhold)
            if (offset != -100):
                result.append(ind + offset)
                ind = ind + offset + 150
                offset = IsEdge(x, ind, errT, threhold)
                if (offset != -100):
                    result.append(ind + offset)
                    ind = ind + offset + 75
                    offset = IsEdge(x, ind, errT, threhold)
                    if (offset != -100):
                        result.append(ind + offset)
                        ind = ind + offset + 225
                        offset = IsEdge(x, ind, errT, threhold)
                        if (offset != -100):
                            result.append(ind + offset)
                            ind = ind + offset + 150
                            offset = IsEdge(x, ind, errT, threhold)
                            if (offset != -100):
                                result.append(ind + offset)
                                return result
    return []


def FindRN16Edges(diff_single_abs):
    dd = np.abs(diff_single_abs)
    tr = np.median(dd) * 5
    # print(tr)
    ind = np.where(dd > tr)[0]
    # print(np.diff(ind))
    result = []
    for i in range(0, len(ind)):
        temp = IsStart(dd, ind[i], 3, tr)
        if (len(temp) > 0):
            # print('havestart')
            result = temp
            break
    if (len(result) == 0):
        return []
    ind = result[len(result) - 1]
    while (ind < len(dd) - 100):
        ind = ind + 75
        offset = IsEdge(dd, ind, 3, tr)
        if (offset != -100):
            ind = ind + offset
            result.append(ind)

    return result


def CalPhaseFromFile(filename, outputfile):
    B = np.fromfile(filename, dtype='float64')
    antennaArray = Add(B[0::4], B[1::4])
    antennaSingle = Add(B[2::4], B[3::4])
    diff_array = np.abs(np.diff(np.abs(antennaArray)))
    # print(np.max(diff_array))
    edge_index = np.where(diff_array > np.max(diff_array) * 0.8)[0]
    # print(diff_array[edge_index])
    array_offset = Counter(np.mod(edge_index, 180)).most_common(1)[0][0]

    single_abs = np.abs(antennaSingle)
    # plt.figure()
    # plt.plot(np.abs(antennaSingle))
    # plt.show()
    index_low = np.where(single_abs <= np.max(single_abs) / 2)[0]
    onelength = np.diff(index_low) - 1

    indexNzero = np.array(np.where((onelength > 0)))[0]
    NoneLength = onelength[indexNzero]
    OneSegStart = index_low[indexNzero]
    OneSegEnd = index_low[indexNzero + 1]
    # print(Counter(NoneLength))
    phaseList = list()
    for i in range(0, 64):
        phaseList.append(list())
    allPhaseList = list()
    allPhaseIndexList = list()
    allPhaseAnt = list()
    allPhaseStrength = list()
    RN16_index = np.where(NoneLength > RN16_length_min)[0]
    for index in range(1):
        start_index = 1000
        end_index = len(single_abs)

        startj = (start_index - array_offset) // 180 + 1
        endj = (end_index - array_offset) // 180 - 1
        for j in range(startj,endj):
            # for kk in range(20,160):
            #     tagphase = np.angle((antennaArray[
            #                                    j * 180 + array_offset +kk] / antennaSingle[
            #                                                                                                j * 180 + array_offset + kk]))
            #
            #     phaseList[int(j) % 64].append(tagphase)
            #     allPhaseList.append(tagphase)
            #     allPhaseIndexList.append(j * 180 + array_offset)
            #     allPhaseAnt.append(j % 64)
            #     allPhaseStrength.append(0)

            phasearray = np.angle((antennaArray[j*180+array_offset+20:j*180+array_offset+160] / antennaSingle[j*180+array_offset+20:j*180+array_offset+160]))
            #phasearray  = np.angle(( antennaSingle[j * 180 + array_offset + 20:j * 180 + array_offset + 160]))
            if np.max(phasearray) - np.min(phasearray) < 6:
                tagphase = np.std(phasearray)
                # tagphase = np.std(np.angle(( antennaSingle[j * 180 + array_offset + 20:j * 180 + array_offset + 160])))

                phaseList[int(j) % 64].append(tagphase)
                allPhaseList.append(tagphase)
                allPhaseIndexList.append(j * 180 + array_offset)
                allPhaseAnt.append(j % 64)
                allPhaseStrength.append(0)


    return allPhaseList, allPhaseIndexList, allPhaseAnt, allPhaseStrength


def AnaOneFolder(folder):
    indd = 1

    start = time.clock()
    output = list()
    for filename in os.listdir(folder):
        if os.path.splitext(filename)[1] == '.bin':
            print(filename)
            pL, pIL, pAL, pSL = CalPhaseFromFile(folder + r'\\' + filename, 'mm' + str(indd) + '.txt')
            t = float(filename.split('_')[0])
            print(t)
            for i in range(len(pL)):
                output.append(str(t + pIL[i] / sampleRate) + ' ' + str(pL[i]) + ' ' + str(pAL[i]) + ' ' + str(pSL[i]))
            indd = indd + 1
    end = time.clock()
    print(end - start)
    with open(folder + r'\\' + 'RFData.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in output)


def AnaMultipleFolders(fatherFolder):
    for foldername in os.listdir(fatherFolder):
        if os.path.isdir(fatherFolder + r'\\' + foldername):
            if os.path.exists(fatherFolder + r'\\' + foldername + r'\\' + 'usrp_data'):
                AnaOneFolder(fatherFolder + r'\\' + foldername + r'\\' + 'usrp_data')



def AnaMultipleFolders2(fatherFolder):
    for foldername in os.listdir(fatherFolder):
        if os.path.isdir(fatherFolder + r'\\' + foldername):
            AnaOneFolder(fatherFolder + r'\\' + foldername)

AnaMultipleFolders(r'D:\Documents\OptiTrack\devices\oqpsk')
# AnaMultipleFolders(r'D:\Documents\OptiTrack\7-12-3')
# AnaMultipleFolders(r'D:\Documents\OptiTrack\7-12-4')
#AnaOneFolder(r'D:\Data\tagargus\lora')
#AnaOneFolder(r'D:\Documents\OptiTrack\7-2-1\5\usrp_data')
# CalPhaseFromFile(r'E:\Data6\2018-12-07-15-52-23-1125461\usrp_data\20181207155432.883_2ch.bin','aaa.txt')
