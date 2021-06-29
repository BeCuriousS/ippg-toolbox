# change dir
# cd /app/shared/research_custom
cd /app/shared/deeplab/SkinDetector1/models-master/research_custom
# add the correct interpreter path
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# remove all data from previous runs
rm /app/shared/data/annotations/*
rm /app/shared/data/information/*
rm /app/shared/data/segmented/*
rm /app/shared/data/segmented_orig/*
rm /app/shared/data/tf_dataset/*
# remove old files from the local destination folder
# NOTE problems when using rm or cp because of "Argument list too long" exception -> therefore using find
find ./deeplab/datasets/ECU_SFA_SCH_HGR/exp/train_on_train_set/vis/ECU_SFA_SCH_HGR,train_rot_gaussian,VOC_trainval,14,False,0.003,0.0,41851,momentum,None,rot_gaussian,6,6/segmentation_results/ -type f -delete
# create the tensorflow dataset
python3 segment_images.py
# start inferencing
bash vis_3_1.sh
# copy result data from local destination to final output dir
find ./deeplab/datasets/ECU_SFA_SCH_HGR/exp/train_on_train_set/vis/ECU_SFA_SCH_HGR,train_rot_gaussian,VOC_trainval,14,False,0.003,0.0,41851,momentum,None,rot_gaussian,6,6/segmentation_results -name "*prediction.png" -exec cp -t /app/shared/data/segmented/ {} +
find ./deeplab/datasets/ECU_SFA_SCH_HGR/exp/train_on_train_set/vis/ECU_SFA_SCH_HGR,train_rot_gaussian,VOC_trainval,14,False,0.003,0.0,41851,momentum,None,rot_gaussian,6,6/segmentation_results -name "*image.png" -exec cp -t /app/shared/data/segmented_orig/ {} +