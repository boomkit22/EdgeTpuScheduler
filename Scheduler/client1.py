from asyncore import write
import os
import time
import random
import threading
from urllib import request

# Model = ['EfficientNet_L', 'EfficientNet_M',EfficientNet_S,'MobileNet_v1']

# Global Variable
FromServer = 0
ToServer = 0
pid = 0
#


def Initialize():
    global ToServer
    global FromServer
    global pid

    pid = os.getpid()
    model = 'EfficientNet_L'
    path = './Pipe/listenPipe'
    msg = '{} {}'.format(str(pid), model).encode()

    listenPipe = os.open(path, os.O_WRONLY)
    os.write(listenPipe, msg)
    os.close(listenPipe)

    ToServerPath = './Pipe/' + str(pid) + 'ToServer_Pp'
    FromServerPath = './Pipe/' + 'ServerTo'+str(pid) + '_Pp'

    while True:
        try:
            FromServer = os.open(FromServerPath, os.O_RDONLY)
            ToServer = os.open(ToServerPath, os.O_WRONLY)
            print('succeed')
            return
        except FileNotFoundError:
            print('failed')
            time.sleep(0.000001)


def Write():
    # 1초에 몇번 보내는지 출력
    global ToServer
    num = 0
    sendPerSec = time.perf_counter()
    ##############
    start = time.perf_counter()
    iter = 0
    period = 0.1
    
    segment_num = 3 
    while True:
        # 주기적으로 서버로 write
        if time.perf_counter() - start > period * iter:
            # img = random.choice(os.listdir("./imageDir"))
            img =  "n0153282900000036.jpg"
            request_time = time.perf_counter()
            args = '{} {} {} {} {}'.format(str(pid), img, str(request_time),str(period), str(segment_num)).encode()
            os.write(ToServer, args)
            print(time.perf_counter())
            num = num + 1
            iter = iter + 1

        #DEBUG #초당 몇번 서버로 write 하는지 print 
        if time.perf_counter() - sendPerSec > 1:
            print(num)
            num = 0
            sendPerSec = time.perf_counter()


def Read():
    global FromServer
    # print('Read From Server {}'.format(FromServer))
    while True:
        readMsg = (os.read(FromServer, 1024)).decode()
        if readMsg:
            print(readMsg)


if __name__ == '__main__':
    Initialize()
    th_Write = threading.Thread(target=Write, args=())
    th_Read = threading.Thread(target=Read, args=())
    th_Write.start()
    th_Read.start()

    th_Write.join()
    th_Read.join()
