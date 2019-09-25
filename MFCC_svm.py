'''
<<<design>>>

1. input 형태 조정 *****
 1.-(1) MFCC 파일을 x, y로 구분하여 리스트에 저장
 (x : 컬럼 차원 리스트, y : 클래스 라벨(1차원 리스트))
2. input을 받아서 svm kernel function을 바꿔서 돌리기
 2-(2) 클래스 여러 개인 부분 조정
 2-(3) kernel function 변경

'''

from sklearn import svm
from sklearn.datasets import make_blobs
from random import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import librosa
import librosa.display
from matplotlib.pyplot import specgram
import os
import IPython.display as ipd

#.wav 파일을 받아와서 MFCC로 변환
def wav2mfcc(path):
    mfccs = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            try:
                X, sample_rate = librosa.load(os.path.join(subdir,file), res_type = "kaiser_fast", duration = 2.5, sr = 22050*2, offset = 0.5)
                mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc = 13)
                y = int(file[7:8])-1
                arr = mfcc, y
                mfccs.append(arr)
                print(mfccs)
            except ValueError:
                continue

    return mfccs

#plot MFCC
def plot_mfcc(mfccfile):
    plt.figure(figsize = (20, 15))
    plt.subplot(3,1,1)
    librosa.display,specshow(mfccfile, x_axis = "time")
    plt.ylabel("MFCC")
    plt.colorbar()

#MFCC csv 파일을  받아와서 리스트에 저장
def readCSVtoList(csvName):
#csvName은 확장자까지 들어간 문자열 데이터
    with open(csvName,'r') as file:
        rawData = file.readlines()
        listData = []
        for data in rawData:
            row = data.split(';')
            listData.append(row)
    return listData


#list를 판다스 df로 보여주기
def print_df(mylist):
    df = pd.DataFrame(data = mylist)
    print(df)


#input 데이터로부터 x데이터 가져오기
def get_x(listData):
    x = []
    x_name = []
    for i in range(1,len(listData)):
        temp = []
        for j in range(len(listData[0])):

            if 'mfcc' in listData[0][j]:
                x_name.append(listData[0][j])

                tv = listData[i][j]
                tv = float(tv)
                temp.append(tv)
        x.append(temp)

    #x = [n_sample, n_features] 형태로 변환
    return x, x_name


#input 데이터로부터 y데이터 가져오기

def get_y(listData):
    y = []
    y_name = 'emotion'
    for i in range(1,len(listData)):
        fileName = listData[i][0]
        fileName = fileName.split('-')
        y.append(int(fileName[2][1]))

    return y

def find_Null(data):
    loca = []
    for i in range(len(data)):
        if isinstance(data[i], list) == True:
            find_Null(data[i])

        if data[i]  == None:
            loca.append(i)

    return loca

def shuffle(x):
    shuffled_list = []

    index_list = []
    while len(x) > len(index_list):
        index = randint(0,len(x)-1)
        if index not in index_list:
            shuffled_list.append(x[index])
            index_list.append(index)

    return shuffled_list

def train_test_split(x, y, test_size):
    temp2 = []
    for c in x:
        temp = shuffle(c)
        temp2.append(temp)
    x = temp2
    y = shuffle(y)
    index = int(len(y) * test_size)
    x_train = []
    x_test = []
    x_train = x[:index]
    x_test = x[index:]
    y_train = y[:index]
    y_test = y[index:]

    return x_train, x_test, y_train, y_test

def accuracy_score(error,total):
    return error / total



'''
-------------------------------------------------
                      main
-------------------------------------------------
'''

#input
'''
listData = readCSVtoList('output.csv') #전역변수랑 로컬변수랑 이름 같아도 되나?
x, x_name = get_x(listData)
y = get_y(listData)
'''
x, y = wav2mfcc("C:\\Users\\PARK\\Documents\\19 summer\\O2DD project\\Audio_Speech_Actors_01-24\\")



#data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print_df(x_train)
print_df(y_train)
print_df(x_test)
print_df(y_test)

#training session
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)

#test session
y_pred = clf.predict(x_test)
print('총 테스트 개수: %d, 오류개수:%d' %(len(y_test), (y_test != y_pred).sum()))
#print('정확도: %.2f' % accuracy_score(y_test,y_pred))

#show result
