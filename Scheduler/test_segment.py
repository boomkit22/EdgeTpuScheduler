import argparse
import re
import threading
import time

import numpy as np
from PIL import Image

from pycoral.adapters import classify
from pycoral.adapters import common
import pycoral.pipeline.pipelined_model_runner as pipeline
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import list_edge_tpus
from pycoral.utils.edgetpu import make_interpreter


def _make_runner(model_paths):
    interpreters = [make_interpreter(m) for m in model_paths]

    for interpreter in interpreters:
        interpreter.allocate_tensors()

    return interpreters


def main():
    labels = read_label_file(
        '/home/hun/WorkSpace/coral/pycoral/test_data/inat_bird_labels.txt')
    
        #     model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_L/efficientnet-edgetpu-L_quant_segment_0_of_3_edgetpu.tflite'
        # model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_L/efficientnet-edgetpu-L_quant_segment_1_of_3_edgetpu.tflite'
        # model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_L/efficientnet-edgetpu-L_quant_segment_2_of_3_edgetpu.tflite'
        # modelName = 'EfficientNet_L'
        # self.model_path_dict[modelName] = [model_1, model_2, model_3]

        # model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_0_of_3_edgetpu.tflite'
        # model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_1_of_3_edgetpu.tflite'
        # model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_2_of_3_edgetpu.tflite'
        # modelName = 'EfficientNet_M'
        # self.model_path_dict[modelName] = [model_1, model_2, model_3]

        # model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_S/efficientnet-edgetpu-S_quant_segment_0_of_3_edgetpu.tflite'
        # model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_S/efficientnet-edgetpu-S_quant_segment_1_of_3_edgetpu.tflite'
        # model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_S/efficientnet-edgetpu-S_quant_segment_2_of_3_edgetpu.tflite'
        # modelName = 'EfficientNet_S'
        # self.model_path_dict[modelName] = [model_1, model_2, model_3]

        # model_1 = '/home/hun/WorkSpace/coral/pycoral/model/segment//Mobilenet_V1/mobilenet_v1_1.0_224_quant_segment_0_of_3_edgetpu.tflite'
        # model_2 = '/home/hun/WorkSpace/coral/pycoral/model/segment//Mobilenet_V1/mobilenet_v1_1.0_224_quant_segment_1_of_3_edgetpu.tflite'
        # model_3 = '/home/hun/WorkSpace/coral/pycoral/model/segment//Mobilenet_V1/mobilenet_v1_1.0_224_quant_segment_2_of_3_edgetpu.tflite'
        # modelName = 'MobileNet_V1'
        # self.model_path_dict[modelName] = [model_1, model_2, model_3]
        

    model_paths = ['/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_0_of_3_edgetpu.tflite',
                   '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_1_of_3_edgetpu.tflite', 
                   '/home/hun/WorkSpace/coral/pycoral/model/segment/Efficient_M/efficientnet-edgetpu-M_quant_segment_2_of_3_edgetpu.tflite']

    interpreter = _make_runner(model_paths)

    # function interpreters()   """Returns list of interpreters that constructed PipelinedModelRunner."""
    size = common.input_size(interpreter[0])   # input _size
    name = common.input_details(interpreter[0], 'name')

    image_path = '/home/hun/WorkSpace/coral/pycoral/examples/Scheduler/imageDir/n0153282900000036.jpg'
    threshold = 0.0
    top_k = 3
    count = 1
    params = common.input_details(
        interpreter[0], 'quantization_parameters')
    scale = params['scales']
    zero_point = params['zero_points']

    mean = 128.0
    std = 128.0
    image_size = Image.open(image_path).size
    print('image size = {}'.format(image_size))
    image = np.array(
        Image.open(image_path).convert('RGB').resize(size, Image.ANTIALIAS))
    ##
    
    
    image_resize_start = time.perf_counter()
    normalized_input = (np.asarray(image) - mean) / \
        (std * scale) + zero_point
    np.clip(normalized_input, 0, 255, out=normalized_input)
    common.set_input(interpreter[0], normalized_input.astype(np.uint8))
    
    image_resize_time = time.perf_counter() - image_resize_start
    print('imaze resize time = {}'.format(image_resize_time*1000))

    print(interpreter[0].get_input_details())
    print('----------------------------------------')
    print(interpreter[0].get_output_details())
    print(interpreter[1].get_input_details())
    print('----------------------------------------')
    print(interpreter[1].get_output_details())
    print(interpreter[2].get_input_details())
    print('---------------------------------------------')
    print(interpreter[2].get_input_details())
    print('--------------------------------------------')
    interpreter
    interpreter[0].invoke()
    print('----------------------------------------')
    print(len(interpreter[0].get_output_details()))
    output_detail_len = len(interpreter[0].get_output_details()) - 1
    tensor_index = interpreter[0].get_output_details()[output_detail_len]['index']
    print(interpreter[0].get_output_details())
    output = interpreter[0].get_tensor(tensor_index)
    common.set_input(interpreter[1], output)
    interpreter[1].invoke()
    # print(interpreter[1].get_output_details())
    print('----------------------------------------')
    # print(len(interpreter[1].get_output_details()))
    print('interpreter 1 output details')
    print(interpreter[1].get_output_details())
    print('interpreter 1 output details')


    print('interpreter 2 input details')
    print(interpreter[2].get_input_details())
    print('interpreter 2 input details')

    # output_detail_len = len(interpreter[1].get_output_details()) - 1
    tensor_index = interpreter[1].get_output_details()[1]['index'] - 1
    print('here = {} '.format(tensor_index))
    output2 = interpreter[1].get_tensor(tensor_index)


    # output3 = interpreter[1].get_tensor(1)
    
    common.set_input(interpreter[2],output2)
    interpreter[2].invoke()
    # output_detail_len = len(interpreter[1].get_output_details()) - 1
    # tensor_index = interpreter[1].get_output_details()[output_detail_len]['index']
    # interpreter[2].invoke()

    classes = classify.get_classes(interpreter[2], top_k, threshold)
    for c in classes:
        print('%s: %.5f\n' % (labels.get(c.id, c.id), c.score))


    
    max_time_invoke = 0
    first_max = 0
    second_max = 0
    third_max = 0
    for i in range(1000):

        first_start = time.perf_counter()

        first_segment_start = time.perf_counter()
        interpreter[0].invoke()
        first_segment_time = time.perf_counter() - first_segment_start
        if first_segment_time > first_max:
            first_max = first_segment_time
        output_detail_len = len(interpreter[0].get_output_details()) - 1
        tensor_index = interpreter[0].get_output_details()[output_detail_len]['index']
        output = interpreter[0].get_tensor(tensor_index)
        common.set_input(interpreter[1], output)
        second_segment_start = time.perf_counter()
        interpreter[1].invoke()
        second_segment_time = time.perf_counter() - second_segment_start
        if second_segment_time > second_max:
            second_max = second_segment_time
        output_detail_len = len(interpreter[1].get_output_details()) - 1
        tensor_index = interpreter[1].get_output_details()[output_detail_len]['index']
        output2 = interpreter[1].get_tensor(tensor_index)
        common.set_input(interpreter[2], output2)
        third_segment_start = time.perf_counter()
        interpreter[2].invoke()
        third_segment_time = time.perf_counter() - third_segment_start
        if third_segment_time > third_max:
            third_max = third_segment_time

        classes = classify.get_classes(interpreter[2], top_k, threshold)
        # for c in classes:
        #     print('%s: %.5f\n' % (labels.get(c.id, c.id), c.score))

        first_time_invoke = time.perf_counter() - first_start
        if first_time_invoke > max_time_invoke :
            max_time_invoke = first_time_invoke
            
        
        
    
    
    print('max time = {}'.format(max_time_invoke * 1000))
    print('first segment max = {}'.format(first_max*1000))
    print('second segment max = {}'.format(second_max*1000))
    print('third segment max = {}'.format(third_max*1000))
    print(first_max * 1000 + second_max * 1000 + third_max * 1000)

    
    ##


    # def producer():
    #     for _ in range(count):
    #         print('push')
    #         runner_first.push({name: image})
    #     runner_first.push({})

    # def consumer():
    #     output_details = runner_second.interpreters()[0].get_output_details()[0]
    #     scale, zero_point = output_details['quantization']
    #     while True:
    #         #Then, receive the pipeline output tensors from PipelinedModelRunner.pop().
    #         #For example, this loop repeatedly accepts new outputs until the result is None:
    #         result = runner_first.pop()
    #         print('runner_first.pop')
    #         if not result:
    #             print('break')
    #             break
    #         values, = result.values()
    #         scores = scale * (values[0].astype(np.int64) - zero_point)
    #         classes = classify.get_classes_from_scores(scores, top_k,
    #                                              threshold)
    #     print('-------RESULTS--------')
    #     for klass in classes:
    #         print('%s: %.5f' % (labels.get(klass.id, klass.id), klass.score))

    # start = time.perf_counter()
    # producer_thread = threading.Thread(target=producer)
    # consumer_thread = threading.Thread(target=consumer)
    # producer_thread.start()
    # consumer_thread.start()
    # producer_thread.join()
    # consumer_thread.join()
    # average_time_ms = (time.perf_counter() - start) / count * 1000
    # print('Average inference time (over %d iterations): %.1fms' %
    #       (count, average_time_ms))


if __name__ == '__main__':
    main()
