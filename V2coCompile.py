# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Example using PyCoral to classify a given image using an Edge TPU.

To run this code, you must attach an Edge TPU attached to the host and
install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
device setup instructions, see coral.ai/docs/setup.

Example usage:
```
bash examples/install_requirements.sh classify_image.py

python3 examples/classify_image.py \
  --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite  \
  --labels test_data/inat_bird_labels.txt \
  --input test_data/parrot.jpg
```
"""

import argparse
from concurrent.futures import process
from inspect import ArgSpec
from sre_constants import SUCCESS
import time
from urllib.request import Request

import numpy as np
import os
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import io
import multiprocessing
from multiprocessing import Process
import threading

# if os.path.exists(S2Cpath):
#   os.remove(S2Cpath)
# if os.path.exists(C2Spath):
#   os.remove(C2Spath)
# if os.path.exists(ImagePath):
#   os.remove(ImagePath)

# os.mkfifo(S2Cpath)
# os.mkfifo(C2Spath)
# os.mkfifo(ImagePath)

#[Pid, imagePath, Model]

# dict  {pid : (modelPath,ToClient)}
#list (pid,imagePath)


manager = multiprocessing.Manager()
request_list = manager.list()
process_dict = manager.dict()
eval_list = manager.list()
# model_interpreter_dict = manager.dict()
# model_interpreter_list = manager.list()

# success = manager.Value('i',0)
# fail = manager.Value('i',0)
# manager.

def ReadWrite(pid, modelName, ServerWritePath, ClientWritePath):
    ToClient = os.open(ServerWritePath, os.O_WRONLY)
    FromClient = os.open(ClientWritePath, os.O_RDONLY)
    
    pid = int(pid)
    newClient = {pid: (modelName, ToClient)}
    process_dict.update(newClient)
    # ToClient = process_dict[pid][1]
    # os.write(ToClient,'asd'.encode())

    while True:
        msg = os.read(FromClient, 100).decode()
        # print(msg)
        pid, imagePath, request_time,period = msg.split(' ')
        pid = int(pid)
        period = float(period)
        request_time = float(request_time)
        if msg:
            request_list.append((pid, imagePath ,request_time, period))


def Listen():
    #os.system('rm ./Pipe/*')
    path = './Pipe/' + 'listenPipe'
    if os.path.exists(path):
        os.remove(path)

    os.mkfifo(path)
    listenPipe = os.open(path, os.O_RDONLY)

    while True:
        readmsg = (os.read(listenPipe, 100)).decode()
        if readmsg:
            pid, modelName = readmsg.split(' ')
  
            
            ServerWritePath = './Pipe/' + 'ServerTo' + str(pid) + '_Pp'
            if os.path.exists(ServerWritePath):
                os.remove(ServerWritePath)
            os.mkfifo(ServerWritePath)

            ClientWritePath = './Pipe/' + str(pid) + 'ToServer_Pp'

            if os.path.exists(ClientWritePath):
                os.remove(ClientWritePath)

            os.mkfifo(ClientWritePath)
            ReadWritePipe = threading.Thread(target=ReadWrite, args=(
                pid, modelName, ServerWritePath, ClientWritePath))
            ReadWritePipe.start()

# RequestNum = 0
success = 0
fail = 0
# prevRequestNum = 0
RequestperSec = 0

def CountRequest():
      global success
      global fail
      # global RequestNum
      # global prevRequestNum
      global RequestperSec
      prevRequestNum = 0

      while True:
        RequestNum = success+fail
        RequestperSec = RequestNum - prevRequestNum
        prevRequestNum = RequestNum
        time.sleep(1)
        print('O : {} X : {} in Queue: {} ReqPerSec : {}'.format(success,fail,len(request_list),RequestperSec))
        
        
#Dictionary 만들어야 함
#model : interpreter
#model 이 존재하지 않으면
#model : interpreter 만듬
def ScheduleAndRun():
    global success
    global fail
    global RequestNum
    global RequestperSec
    Count_Request = threading.Thread(target = CountRequest, args = ())
    Count_Request.start()

    #######   Initialize   ######
              
              
    # Model = ['EfficientNet_L', 'EfficientNet_M',EfficientNet_S,'MobileNet_v1']
    
    #요청을 받으면 모델과 interpreter를 만들어서
    model_path_list = []
    model_interpreter_dict = {}
    model = '/home/hun/WorkSpace/coral/pycoral/model/model_set2/efficientnet-edgetpu-L_quant_edgetpu.tflite'
    modelName = 'EfficientNet_L'
    model_path_list.append((modelName,model))

    model = '/home/hun/WorkSpace/coral/pycoral/model/model_set2/efficientnet-edgetpu-M_quant_edgetpu.tflite'
    modelName = 'EfficientNet_M'
    model_path_list.append((modelName,model))

    model = '/home/hun/WorkSpace/coral/pycoral/model/model_set2/efficientnet-edgetpu-S_quant_edgetpu.tflite'
    modelName = 'EfficientNet_S'
    model_path_list.append((modelName,model))
    
    model = '/home/hun/WorkSpace/coral/pycoral/model/model_set2/mobilenet_v1_0.25_128_quant_edgetpu.tflite'
    modelName = 'MobileNet_V1'
    model_path_list.append((modelName,model))
       
       
    for model in model_path_list:   
        Interpreter = make_interpreter(model[1])
        Interpreter.allocate_tensors()
        newInterpreter = { model[0] : Interpreter }
        model_interpreter_dict.update(newInterpreter)
        
    
        # model_interpreter_list.append(newInterpreter)
    #dictionary에 추가
#  if modelName == 'EfficientNet_S':
#     elif modelName == 'MobileNetV3':
#     elif modelName == 'ResNet':
        
    #######                ####### 
    while True:
        if len(request_list) != 0:
            loopStartTime = time.perf_counter()
            
            # print(len(request_list))
            who = request_list.pop(0)
            pid = who[0]
            imagePath = './imageDir/' + who[1]
            request_time = who[2]
            period = who[3]
            deadline = period
            
            modelName = process_dict[pid][0]
            ToClient = process_dict[pid][1]           
            #요청을 받으면 모델과 interpreter를 만들어서
            interpreter = model_interpreter_dict[modelName]
                       
            # Interpreter = make_interpreter(model)
            # Interpreter.allocate_tensors()
            #dictionary에 추가

            count = 5
            threshold = 0.0
            top_k = 3

            
            labels = read_label_file(
                '/home/hun/WorkSpace/coral/pycoral/test_data/labels.txt')

            if common.input_details(interpreter, 'dtype') != np.uint8:
                raise ValueError('Only support uint8 input type.')

            size = common.input_size(interpreter)
            image = Image.open(imagePath).convert(
                'RGB').resize(size, Image.ANTIALIAS)
            params = common.input_details(
                interpreter, 'quantization_parameters')
            scale = params['scales']
            zero_point = params['zero_points']

            mean = 128.0
            std = 128.0
            if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
                # Input data does not require preprocessing.
                common.set_input(interpreter, image)
            else:
                # Input data requires preprocessing
                normalized_input = (np.asarray(image) - mean) / \
                    (std * scale) + zero_point
                np.clip(normalized_input, 0, 255, out=normalized_input)
                common.set_input(
                    interpreter, normalized_input.astype(np.uint8))

            # Run inference
            # print('----INFERENCE TIME----')
            # print('Note: The first inference on Edge TPU is slow because it includes',
            #       'loading the model into Edge TPU memory.')
            for _ in range(1):
                start = time.perf_counter()
                interpreter.invoke()
                inference_time = time.perf_counter() - start
                classes = classify.get_classes(interpreter, top_k, threshold)
                # print('%.1fms' % (inference_time * 1000))

            # print('-------RESULTS--------')
            msg = ''
            for c in classes:
                #print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
                msg = msg + ('%s: %.5f\n' % (labels.get(c.id, c.id), c.score))

            # print(ToClient)
            # with os.open(ServerWritePath, os.O_WRONLY) as ToClient:
            # print(msg)
            os.write(ToClient,msg.encode())
            if time.perf_counter() - request_time < deadline:
                  # print(time.perf_counter() - request_time)
                  success = success+1
            else:
                  fail = fail + 1
            
            # print('O : {} X : {} in Queue: {} ReqPerSec : {}'.format(success,fail,len(request_list),RequestperSec))
            # print('while loop time = {}ms'.format((time.perf_counter() - loopStartTime)*1000))

            

            


if __name__ == '__main__':
    os.system('rm ./Pipe/*')
    # listenProcess = Process(target=Listen, args=())
    # scheduler = Process(target=ScheduleAndRun, args=())
    listenProcess = threading.Thread(target=Listen, args=())
    scheduler = threading.Thread(target=ScheduleAndRun, args=())

    listenProcess.start()
    scheduler.start()

    listenProcess.join()
    scheduler.join()
