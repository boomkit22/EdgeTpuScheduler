from asyncore import write
import os
import time
import random
from urllib import request

# from coral.pycoral.examples.client import C2Spipe


#EfficientNEt-EdgeTpu(L)
#MobileNet V1
#MobileNet V3

Model = ['EfficientNet_S', 'MobileNetV3','ResNet']


############Initialize#############
pid = os.getpid();
model = 'EfficientNet_S'
path = './Pipe/listenPipe'
msg = '{} {}'.format(str(pid), model).encode()

listenPipe = os.open(path, os.O_WRONLY)
os.write(listenPipe,msg)
os.close(listenPipe)
############Initialize#############



############## Send Random Image To Server ################
ToServer =  './Pipe/' + str(pid) + 'ToServer_Pp'
FromServer = './Pipe/' + 'ServerTo'+str(pid) +'_Pp'


# Open Pipe until Server made Pipe
while True:
    try:
        FromServer = os.open(FromServer,os.O_RDONLY)
        ToServer = os.open(ToServer, os.O_WRONLY)
        print('succeed')
        break
    except FileNotFoundError:
        print('failed')
        time.sleep(0.000001)

#Send Random Image to Server
while True:
    img = random.choice(os.listdir("./imageDir"));
    request_time = time.perf_counter()
    args = '{} {} {}'.format(str(pid), img, str(request_time)).encode()
    # print(args)
    os.write(ToServer, args)
    readmsg = (os.read(FromServer, 1024)).decode()
    if readmsg:
        print(readmsg)
    time.sleep(0.05)

############## Randomly Send Image To Server ################







# img = random.choice(os.listdir("/home/mendel/mini-imagenet/images"));


# print(img)
# msg = img.encode()
# print(msg)
# print(msg.decode())

# model = random.choice(Model)
# print(model)

# args = '{} {}'.format(img,model).encode()
# print(args)
