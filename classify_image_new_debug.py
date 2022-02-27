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
from concurrent.futures import process, thread
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

import threading




lock = threading.Lock()
request_list = []
process_dict = {}

def ReadWrite(pid, modelName, ServerWritePath, ClientWritePath):
    global process_dict
    global request_list
    
    ToClient = os.open(ServerWritePath, os.O_WRONLY)
    FromClient = os.open(ClientWritePath, os.O_RDONLY)
    
    pid = int(pid)
    process_dict[pid] = (modelName, ToClient)

    while True:
        msg = os.read(FromClient, 100).decode()
        try:
            pid, imagePath, request_time,period = msg.split(' ')
        except(ValueError):
            print(msg)
            
        pid = int(pid)
        period = float(period)
        request_time = float(request_time)
        if msg:
            lock.acquire()
            request_list.append((pid,imagePath,request_time,period))
            lock.release()

def Listen():
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
        # print('O : {} X : {} in Queue: {} ReqPerSec : {}'.format(success,fail,len(request_list),RequestperSec))
        
        

def ScheduleAndRun():
    
    global success
    global fail
    global RequestNum
    global RequestperSec
    
    global request_list
    global process_dict
    
    Count_Request = threading.Thread(target = CountRequest, args = ())
    Count_Request.start()

    #######   Initialize   ######
              
              
    # Model = ['EfficientNet_L', 'EfficientNet_M',EfficientNet_S,'MobileNet_v1']  
    model_interpreter_dict = {}
    model_path_list = []
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
       
    labels = read_label_file('/home/hun/WorkSpace/coral/pycoral/test_data/labels.txt')  
    
    for model in model_path_list:   
        Interpreter = make_interpreter(model[1])
        Interpreter.allocate_tensors()
        model_interpreter_dict[model[0]] = Interpreter
        
    #######                ####### 
    while True:
        if len(request_list) != 0:
            loopStartTime = time.perf_counter()          
            # print(len(request_list))
            ##FIFO##
            lock.acquire()
            request_list.sort(key = lambda request: request[3])
            who = request_list.pop(0)
            lock.release()
            ########            
            pid = who[0]
            imagePath = './imageDir/' + who[1]
            request_time = who[2]
            period = who[3]
            #print(period)
            deadline = period
        
            modelName = process_dict[pid][0]
            ToClient = process_dict[pid][1]           
            #요청을 받으면 모델과 interpreter를 만들어서
            interpreter = model_interpreter_dict[modelName]

            count = 5
            threshold = 0.0
            top_k = 3

            schedulingOverheadTime = time.perf_counter() - loopStartTime
          
            
            
            if common.input_details(interpreter, 'dtype') != np.uint8:
                raise ValueError('Only support uint8 input type.')

            size = common.input_size(interpreter)
            # image = Image.open(imagePath).convert('RGB')
            # print(image.size)
            
            imageResizeStart = time.perf_counter()
            image = Image.open(imagePath).convert(
                'RGB').resize(size, Image.ANTIALIAS)
            imageResizeTime = time.perf_counter() - imageResizeStart
            
            params = common.input_details(
                interpreter, 'quantization_parameters')
            scale = params['scales']
            zero_point = params['zero_points']

            mean = 128.0
            std = 128.0
            
          
            
            
            setInputStart = time.perf_counter()
            if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
                # Input data does not require preprocessing.
                preprocessingFlag = 0
                common.set_input(interpreter, image)
            else:
                # Input data requires preprocessing
                preprocessingFlag = 1
                normalized_input = (np.asarray(image) - mean) / \
                    (std * scale) + zero_point
                np.clip(normalized_input, 0, 255, out=normalized_input)
                common.set_input(
                    interpreter, normalized_input.astype(np.uint8))
            setInputTime  = time.perf_counter() - setInputStart
            # Run inference
            # print('----INFERENCE TIME----')
            # print('Note: The first inference on Edge TPU is slow because it includes',
            #       'loading the model into Edge TPU memory.')
            tpuInvokeStart = time.perf_counter()
            
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
                
                
            tpuInvokeTime = time.perf_counter() - tpuInvokeStart
            os.write(ToClient,msg.encode())
            
            
            if time.perf_counter() - request_time < deadline:
                  # print(time.perf_counter() - request_time)
                  success = success+1
            else:
                  fail = fail + 1
            
            # print('O : {} X : {} in Queue: {} ReqPerSec : {}'.format(success,fail,len(request_list),RequestperSec))
            print('loop time = {:.3f}ms'.format((time.perf_counter() - loopStartTime)*1000))
            if preprocessingFlag:
                print('requires preprocessing')
            else:
                print('non preprocessing')
            print('schedulingOverheadTime = {:.3f}ms'.format((schedulingOverheadTime)*1000))
            print('imageResizeTime = {:.3f}ms'.format((imageResizeTime)*1000))
            print('setInputTime = {:.3f}ms'.format((setInputTime)*1000))
            print('tpuInvokeTime = {:.3f}ms'.format((tpuInvokeTime)*1000))
            print('remain time = {:.3f}ms'.format((time.perf_counter() - loopStartTime - schedulingOverheadTime - imageResizeTime
                                                 - tpuInvokeTime - setInputTime)*1000))
            

            


if __name__ == '__main__':
    os.system('rm ./Pipe/*')

    listenProcess = threading.Thread(target=Listen, args=())
    scheduler = threading.Thread(target=ScheduleAndRun, args=())

    listenProcess.start()
    scheduler.start()

    listenProcess.join()
    scheduler.join()
