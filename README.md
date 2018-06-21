Tensorflow-gpu on Mac High Sierra(10.13.5) configuration with Python2.7
========
#### Mac external GPU with CUDA support

I have been working on the configuration for two weeks... the first week was because I forgot to connect the PCI-e wire to my GPU card like this:
<img src="https://github.com/wyyqwqq/Tensorflow-gpu-on-MAC-High-Sierra-10.13.5-with-Python-2.7/blob/master/IMG_2709.JPG" width="350">

so if you met some problems like 'disconnect "null"' or "Nvidia Chip Model" in "System report", check your connection and reboot your Mac with eGPU connected. Then the second week was because the compilation of tensorflow.<br>
I hope this doc can help you finish your configuration as soon as possible.

## Environment
  `Check your environment before you start`
  * OSX: High Sierra 10.13.5
  * MacBook Pro (13-inch, 2016, Two Thunderbolt 3 ports)
  * Nvidia GForce 1060 6G
  * Python: 2.7


## Background Environment

### 1. Install GPU Driver(Nvidia Web Driver-387.10.10.10.35.106)
  You can download the newest driver from here:  [Download GPU Driver](http://www.nvidia.com/download/driverResults.aspx/134834/en-us)<br>
  the version is 387.10.10.10.35.106. If you install the wrong version, it will probably not recognize your eGPU.<br>

 
### 2. Install CUDA v9.1
  You can download CUDA 9.1 from here:  [Download CUDA 9.1](https://developer.nvidia.com/cuda-downloads?target_os=MacOSX&target_arch=x86_64&target_version=1013&target_type=dmglocal)<br>
  Don't install CUDA version lower than 9.1 if your OSX version >= 10.13.5, otherwise it can not find your eGPU.<br>
  I know that [Tensorflow-gpu](https://www.tensorflow.org/versions/r1.1/install/install_mac) for Mac require you to install CUDA 8.0, but it's not the case for eGPU with High Sierra.
  
  
### 3. Install cuDNN v7.0.5
  You can download cuDNN from here: [Download cuDNN v7.0.5 for CUDA 9.1](https://developer.nvidia.com/rdp/cudnn-archive)<br>
  Then use the following code in Terminal to install it:
    
    tar -xzvf cudnn-9.1-osx-x64-v7-ga.tgz
    sudo cp cuda/include/cudnn.h /usr/local/cuda/include
    sudo cp cuda/lib/libcudnn* /usr/local/cuda/lib
    sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib/libcudnn*
    export  DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
  
***If you don't need your eGPU work for deep learning, you're good to go!***
  
  
### 4. Install Xcode Command Line Tool 8.3.2 or 8.2
  You can download Command Line Tool from here:  [Download Command line tool](https://developer.apple.com/download/more/)<br>
  1. Install it<br>
  2. Then switch to it:
    
    sudo xcode-select --switch /Library/Developer/CommandLineTools
  3. verify it: 
    
    clang -v


### 5. Setup environment variables
  In Terminal, run following command:<br>
    
    export CUDA_HOME=/usr/local/cuda
    export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:/usr/local/cuda/extras/CUPTI/lib
    export LD_LIBRARY_PATH=$DYLD_LIBRARY_PATH
    export PATH=$DYLD_LIBRARY_PATH:$PATH:/Developer/NVIDIA/CUDA-9.1/bin
  or you can add them in to .bash file


### 6. Check your CUDA installation
  Run following code in Terminal:<br>
    
    cd /Developer/NVIDIA/CUDA-9.1/samples
    sudo make -C 1_Utilities/deviceQuery
    ./Developer/NVIDIA/CUDA-9.1/samples/bin/x86_64/darwin/release/deviceQuery
  
  You will see following results if your CUDA installation is successful:
    
    Detected 1 CUDA Capable device(s)
    
    Device 0: "GeForce GTX 1060 6GB"
    CUDA Driver Version / Runtime Version          9.2 / 9.1
    CUDA Capability Major/Minor version number:    6.1
    ......
  If it can find your eGPU, then you can start to build your Tensorflow<br>



## Build Tensorflow

### 7. Install Wheel
  Run following code in Terminal:<br>
    
    pip install wheel


### 8. Install Bazel 0.10
  [Download Bazel 0.10 here](https://github.com/bazelbuild/bazel/releases/download/0.10.0/bazel-0.10.0-installer-darwin-x86_64.sh)<br>
  Recommend this version, otherwise you may get tons of weird errors during compilation.<br>
  Run following code in Terminal to install:
    
    chmod 755 bazel-0.10.0-installer-darwin-x86_64.sh
    ./bazel-0.10.0-installer-darwin-x86_64.sh
  

### 9. Git clone Tensorflow 1.7 to your folder
  `Recommend this version, because 1.8 or higher may cause tons of weird errors during compilation.`<br>
  Run following code in Terminal:
    
    git clone https://github.com/tensorflow/tensorflow
    cd tensorflow
    git checkout v1.7.0
  Then download a patch to current folder:
    
    wget https://gist.githubusercontent.com/Willian-Zhang/088e017774536880bd425178b46b8c17/raw/xtensorflow17macos.patch
    git apply xtensorflow17macos.patch
  
  
### 10. Configuration 
  Run following code under current folder in Terminal:<br>
    
    ./configure
  You will get following info:
    
    You have bazel 0.10.0 installed:
    
    Please specify the location of python. \[Default is /usr/bin/python]: 

    Found possible Python library paths:
      /Library/Python/2.7/site-packages
    Please input the desired Python library path to use.  Default is [/Library/Python/2.7/site-packages]

    Do you wish to build TensorFlow with Google Cloud Platform support? [Y/n]: n
    No Google Cloud Platform support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with Hadoop File System support? [Y/n]: n
    No Hadoop File System support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with Amazon S3 File System support? [Y/n]: n
    No Amazon S3 File System support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with Apache Kafka Platform support? [y/N]: n
    No Apache Kafka Platform support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with XLA JIT support? [y/N]: n
    No XLA JIT support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with GDR support? [y/N]: n
    No GDR support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with VERBS support? [y/N]: n
    No VERBS support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: n
    No OpenCL SYCL support will be enabled for TensorFlow.

    Do you wish to build TensorFlow with CUDA support? [y/N]: y
    CUDA support will be enabled for TensorFlow.

    Please specify the CUDA SDK version you want to use, e.g. 7.0. [Leave empty to default to CUDA 9.0]: 9.1

    Please specify the location where CUDA 9.1 toolkit is installed. Refer to README.md for more details. [Default is  /usr/local/cuda]: 

    Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 

    Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:

    Please specify a list of comma-separated Cuda compute capabilities you want to build with.
    You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
    Please note that each additional compute capability significantly increases your build time and binary size. [Default is: 3.5,5.2]6.1

    Do you want to use clang as CUDA compiler? [y/N]: n
    nvcc will be used as CUDA compiler.

    Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]: 

    Do you wish to build TensorFlow with MPI support? [y/N]: n
    No MPI support will be enabled for TensorFlow.

    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native]: 

    Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: n
    Not configuring the WORKSPACE for Android builds.

    Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See tools/bazel.rc for more details.
      --config=mkl         	# Build with MKL support.
      --config=monolithic  	# Config for mostly static monolithic build.
    Configuration finished
    
  
### 11. Compilation Tensorflow
  Run following code in Terminal:
    
    bazel build --config=cuda --config=opt --action_env PATH --action_env LD_LIBRARY_PATH --action_env DYLD_LIBRARY_PATH //tensorflow/tools/pip_package:build_pip_package
  Takes me 1 hour to finish.
  
  
### 12. Build and install Tensorflow wheel file
  Run the following code in Terminal:
    
    bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/
  Install Tensorflow, ***if you prefer using virtualenv, then activate your virtualenv and run the following code\:*** <br>
    
    pip install ~/tensorflow-1.7.0-cp27-cp27m-macosx_10_13_intel.whl (Whataver filename and path you have based on different environment)
  
  
### 13. Verify
  `You're almost done!!!!!!!!!`<br>
  Open Python in virtualenv, and run following code in Terminal:<br>
    
    >>>import tensorflow as tf
    >>>hello = tf.constant('hello')
    >>>sess = tf.Session()
    >>>print(sess.run(hello))
  
  `If there's no error, then you're done!!!!`
  You can see some info about your eGPU:
      
      Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 880 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1060 6GB, pci bus id: 0000:85:00.0, compute capability: 6.1)
  
  
  
## Reference
 >https://gist.github.com/Willian-Zhang/088e017774536880bd425178b46b8c17
 >https://gist.github.com/pavelmalik/d51036d508c8753c86aed1f3ff1e6967
 >https://www.tensorflow.org/versions/r1.1/install/install_mac
