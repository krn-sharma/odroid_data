
import pyarmnn as ann
import numpy as np
import argparse
import cv2
import json
import os
from timeit import default_timer as timer

print(f"Working with ARMNN {ann.ARMNN_VERSION}")
# ONNX, Caffe and TF parsers also exist.
parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile('./sliding.tflite')

graph_id = 0
input_names = parser.GetSubgraphInputTensorNames(graph_id)
input_binding_info = parser.GetNetworkInputBindingInfo(graph_id, input_names[0])
input_tensor_id = input_binding_info[0]
input_tensor_info = input_binding_info[1]
print(f"""
tensor id: {input_tensor_id},
tensor info: {input_tensor_info}
""")

# Create a runtime object that will perform inference.
options = ann.CreationOptions()
runtime = ann.IRuntime(options)

# Backend choices earlier in the list have higher preference.
#preferredBackends = [ann.BackendId('GpuAcc')]
preferredBackends = [ann.BackendId('CpuAcc')]
opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

# Load the optimized network into the runtime.
net_id, _ = runtime.LoadNetwork(opt_network)
print(f"Loaded network, id={net_id}")

dir = "/home/odroid/Documents/ML-examples/DL/MAIZE/karan_results"
for rmfile in os.listdir(dir):
  os.remove(os.path.join(dir, rmfile))

d2 = "/home/odroid/Documents/ML-examples/DL/MAIZE/data/Test PVD"
total_correct = 0; total_img = 0;

labels = ['background', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Corn_(maize)___nutrient_deficient']
for f2 in os.listdir(d2):
  ##f2( Eg. NLB)
  d3 = os.path.join(d2,f2)
  total_in_class = 0; correct_in_class = 0;
  start = timer()
  for images in os.listdir(d3):
    ##images( Eg. test_5087.JPEG)
    if(1):
      img_loc = os.path.join(d3,images)
      image = cv2.imread(img_loc)
      image = cv2.resize(image, (740, 740))
      image = np.array(image, dtype=np.float32) / 255.0
      #print(image.shape)
      total_in_class+=1
      total_img+=1
      # Create an inputTensor for inference.
      input_tensors = ann.make_input_tensors([input_binding_info], [image])

      # Get output binding information for an output layer by using the layer name.
      output_names = parser.GetSubgraphOutputTensorNames(graph_id)
      output_binding_info = parser.GetNetworkOutputBindingInfo(0, output_names[0])
      output_tensors = ann.make_output_tensors([output_binding_info])

      #cmd = "taskset -c 0,1,2,3 python3 -c runtime.EnqueueWorkload(0, input_tensors, output_tensors)"
      #start = timer()
      #os.system(cmd)
      runtime.EnqueueWorkload(0, input_tensors, output_tensors)
      #end = timer()
      #print('Elapsed time is ', (end - start) * 1000, 'ms')

      preds = ann.workload_tensors_to_ndarray(output_tensors)
      #print(f"Output tensor info: {result}")
      #print(labels[np.argmax(preds[0][0])])
      #preds = model.predict(x)

      
      img = images[:-4]
      name = "/home/odroid/Documents/ML-examples/DL/MAIZE/karan_results/"+f2+".txt"
      ff = open(name,'a')
      
      #print(values)
      ff.write(img)
      ff.write(",")
      ff.write(labels[0])
      ff.write(" ")
      ff.write(str(preds[0][0][0]))
      ff.write(",")
      ff.write(labels[1])
      ff.write(" ")
      ff.write(str(preds[0][0][1]))
      ff.write(",")
      ff.write(labels[2])
      ff.write(" ")
      ff.write(str(preds[0][0][2]))
      ff.write(",")
      ff.write(labels[3])
      ff.write(" ")
      ff.write(str(preds[0][0][3]))
      ff.write(",")
      ff.write(labels[4])
      ff.write(" ")
      ff.write(str(preds[0][0][4]))
      ff.write("\n")
      ff.close()


      #print(labels[np.argmax(preds[0][0])]," ",f2)
      #if(labels[np.argmax(preds[0][0])] == f2):
       # correct_in_class+=1
      #  total_correct+=1
      print(preds)
      one = np.zeros((18,18))
      #for i in range(18):
      	#for j in range(18):

      for i in range(18):
        for j in range(18):
	  m = np.where(output_data[0][i][j]==np.max(output_data[0][i][j])) #
	  one[i][j] = m[0]
      print(one)
      unique, counts = np.unique(one, return_counts=True)
      print(dict(zip(unique, counts)))
      exit()
      if(f2 == "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot" or f2=="Corn_(maize)___Common_rust_" or f2=="Corn_(maize)___Northern_Leaf_Blight"):
        olab = labels[np.argmax(preds[0][0])]
        if(olab == "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot" or olab=="Corn_(maize)___Common_rust_" or olab=="Corn_(maize)___Northern_Leaf_Blight"):
          correct_in_class+=1
          total_correct+=1
          #print("inc")
      else:
        if(labels[np.argmax(preds[0][0])] == f2):
          correct_in_class+=1
          total_correct+=1
          #print("inc")
      
  end = timer()
  t = (end - start)
  print(f2," Top1: ",correct_in_class,"/",total_in_class," Time Taken: ",t," seconds")
print("Error Top1 : ",1-(total_correct/total_img))
