# Instructions to run deeplab for skin detection

- The deeplab implementation is not easy to use at the moment as it is based on the original repository of deeplab: https://github.com/tensorflow/models/tree/master/research/deeplab

## Steps to use this repository with your dataset

- Create the docker container with the dockerfile within this folder
  - ```shell´´´
    cd ippgtoolbox/detection/skin/deeplab
    docker build -t ippg-toolbox-deeplab .
    ```
- Create your own processing files for YourDataset based on BP4D+ or UBFC dataset processing:
  - _run_deepLab_on_YourDataset.sh_
  - _dockerrun_YourDataset.sh_
  - _prepare_images_YourDataset.sh_
- Create a directory that can be used for file exchange with the following structure:
  - _YourTmpPath/tmp_ (base directory)
  - _YourTmpPath/tmp/segmented_ (sub directory)
  - _YourTmpPath/tmp/segmented_orig_ (sub directory
  - _YourTmpPath/tmp/orig_ (sub directory)
  - _YourTmpPath/tmp/orig/original_ (sub directory)
  - _YourTmpPath/tmp/orig/resized_ (sub directory)
  - _YourTmpPath/tmp/orig/face_detection_info_ (sub directory)
- Download the _deeplab_model_data_:
  - ```shell´´´
    cd ippgtoolbox/detection/skin/deeplab
    wget https://cloudstore.zih.tu-dresden.de/index.php/s/S4AoF6rWKrEGASE/download
    ```
- Unpack the _deeplab_model_data_:
  - ```shell´´´
    cd ippgtoolbox/detection/skin/deeplab
    unzip download -d "SkinDetector1/models-master/research_custom/deeplab/datasets/ECU_SFA_SCH_HGR/exp/train_on_train_set/train/ECU_SFA_SCH_HGR,train_rot_gaussian,VOC_trainval,14,False,0.003,0.0,41851,momentum,None,rot_gaussian,6,6"
    ```

## File description

- _run_deepLab_on_YourDataset.sh_
  - Main file to start the docker container with the necessary commands
  - You need to adjust:
    - _SRC_ contains one folder for each video recording and each video recording consists of images
    - _DST_ represents the destination folder where the segmented images (output of deeplab) should be saved
    - _TMP_ represents the destination folder of the segmented images. Set this as follows: _YourTmpPath/tmp/segmented_ (this path will be created automatically)
    - _INFO_ represents the destination path for the information about the position of the detected face
    - Line 25 to call the correct script _dockerrun_YourDataset.sh_
- _dockerrun_YourDataset.sh_
  - File to start the docker container (is called by _run_deepLab_on_YourDataset.sh_)
  - You need to adjust:
    - The directories in line 9 to 11 to the temporary path you set before (_YourTmpPath/tmp/orig/..._)
    - The directory in line 15 pointing to the folder containing the images of the video recording
    - The directory in line 15 pointing to the temporary path you set before (_YourTmpPath/tmp/orig/original_)
    - Line 16 to call the correct image preparation script (_prepare_images_YourDataset.py_)
    - The volumes to bind when starting the docker container:
      - line 22: the temporary path (_YourTmpPath/tmp_)
      - line 23: the absolute path to the deeplab repository location (within this repository)
- _prepare_images_YourDataset.py_
  - Adjust this file to your needs. It must do the following:
    - Create resized (512x512 pixels) images which are then used as the input for deeplab
    - The images must be saved in the temporary directory (_YourTmpPath/orig/resized_) which will then be accessed by deeplab

## Run

- When adjusted the previous mentioned files you can run deeplab on your dataset executing _run_deepLab_on_YourDataset.sh_ with:

```shell´´´
cd path/to/folder/run_deepLab_on_YourDataset.sh
bash run_deepLab_on_YourDataset.sh
```

## Test deeplab

- Do the following steps from _Steps to use this repository with your dataset_:
  - Download the _deeplab_model_data_
  - Unpack the _deeplab_model_data_
- Open a terminal
- Change working directory to _ippgtoolbox/detection/skin/deeplab/tests_
- Run \_run_deepLab_on_testDataset.sh
