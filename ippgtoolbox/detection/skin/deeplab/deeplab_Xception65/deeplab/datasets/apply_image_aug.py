import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import cv2
import os
import imageio
import random


# In this script the image augmentation is applied. Here, the images are rotated with 90, 180, 270 degrees, additive gaussian is applied and the brightness is adjusted
# The augmented images are named in accordance to their augmentation. On top, the txt files containing all the image names are created.
# The annotations are not augmented, but duplicated for the augmented images, so each augmented image has its annotation for training





path = os.path.dirname(os.path.realpath(__file__))
path_images = path + "/ECU_SFA_SCH_HGR/JPGImages" # unaugmented image source
path_annotations = path + "/ECU_SFA_SCH_HGR/SegmentationClass_ign" # unaugmented annotation source
path_aug_images = path + "/ECU_SFA_SCH_HGR/JPGImages_aug" # folder where the augmented images are stored
path_aug_annotations = path + "/ECU_SFA_SCH_HGR/SegmentationClass_ign_aug" # folder where the annotations of the augmented images are stored
print("this is the path", path)


original_files = [os.path.join(root, name)
            for root, dirs, files in os.walk(path_images)
            for name in files
            if name.endswith(".jpg")]
annotation_files = [os.path.join(root, name)
            for root, dirs, files in os.walk(path_annotations)
            for name in files
            if name.endswith(".png")]
        


def augment_image(angle,image_path):
    img = np.array(imageio.imread(image_path)) #read image
    seq = iaa.Sequential(
                        [
                            iaa.Rotate(rotate=angle)
                        ])
    image_aug = seq.augment_image(img)
    if angle == 0:
        imageio.imwrite(path_aug_images + "/" + image_path.split("/")[-1], images_aug)
    elif image_path.endswith('jpg'):
        imageio.imwrite(path_aug_images + "/" + image_path.split("/")[-1].split(".jpg")[0] + "_rot{}".format(angle) + ".jpg", image_aug)
    elif image_path.endswith('png'):
        imageio.imwrite(path_aug_images + "/" + image_path.split("/")[-1].split(".png")[0] + "_rot{}".format(angle) + ".png", image_aug)

def rotate_image(angle,image_path):
    img = cv2.imread(image_path) #read image
    if angle == 90:
        image_aug = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        image_aug = cv2.rotate(img,cv2.ROTATE_180)
    elif angle == 270:
        image_aug = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 0 and image_path.endswith('jpg'):
        cv2.imwrite(path_aug_images + "/" + image_path.split("/")[-1], img)
    elif angle == 0 and image_path.endswith('png'):
        cv2.imwrite(path_aug_annotations + "/" + image_path.split("/")[-1], img)
    elif image_path.endswith('jpg'):
        cv2.imwrite(path_aug_images + "/" + image_path.split("/")[-1].split(".jpg")[0] + "_rot{}".format(angle) + ".jpg", image_aug)
    elif image_path.endswith('png'):
        cv2.imwrite(path_aug_annotations + "/" + image_path.split("/")[-1].split(".png")[0] + "_rot{}".format(angle) + ".png", image_aug)


def noise_brightness(i,multiply,gauss,image_path):
    img = np.array(imageio.imread(image_path)) #read image
    ia.seed(i)

    seq = iaa.Sequential(
        [     
            iaa.Multiply((multiply), per_channel=False),  
 
            #iaa.OneOf([  
                #iaa.AdditiveGaussianNoise(
                    #loc=0, scale=(0.04 * 255), per_channel=True),    
                #iaa.AdditiveLaplaceNoise(scale=(0.04*255), per_channel=True),
                #iaa.AdditivePoissonNoise((0.04*255), per_channel=True),
            #])
        ])
    images_aug = seq.augment_image(img)
    imageio.imwrite(path_aug_images + "/" + image_path.split("/")[-1].split(".jpg")[0] + "_mult_{}".format(multiply) + ".jpg", images_aug)

    seq = iaa.Sequential(
        [     
            iaa.AdditiveGaussianNoise(
                    loc=0, scale=(gauss * 255), per_channel=True),  
 
            #iaa.OneOf([  
                #iaa.AdditiveGaussianNoise(
                    #loc=0, scale=(0.04 * 255), per_channel=True),    
                #iaa.AdditiveLaplaceNoise(scale=(0.04*255), per_channel=True),
                #iaa.AdditivePoissonNoise((0.04*255), per_channel=True),
            #])
        ])
    images_aug = seq.augment_image(img)
    imageio.imwrite(path_aug_images + "/" + image_path.split("/")[-1].split(".jpg")[0] + "_gaussian_{}".format(gauss) + ".jpg", images_aug)


    seq = iaa.Sequential(
        [     
            iaa.AdditiveGaussianNoise(
                    loc=0, scale=(gauss * 255), per_channel=True),  
            iaa.Multiply((multiply), per_channel=False),  
 
            #iaa.OneOf([  
                #iaa.AdditiveGaussianNoise(
                    #loc=0, scale=(0.04 * 255), per_channel=True),    
                #iaa.AdditiveLaplaceNoise(scale=(0.04*255), per_channel=True),
                #iaa.AdditivePoissonNoise((0.04*255), per_channel=True),
            #])
        ])
    images_aug = seq.augment_image(img)
    imageio.imwrite(path_aug_images + "/" + image_path.split("/")[-1].split(".jpg")[0] + "_gaussian_{}_mult_{}".format(gauss,multiply) + ".jpg", images_aug)


    
geometrical_transformation = False
if geometrical_transformation == True:
    for image_path in original_files:
        rotate_image(0,image_path)
        rotate_image(90,image_path)
        rotate_image(180,image_path)
        rotate_image(270,image_path)
    print('finished images')

    for image_path in annotation_files:
        rotate_image(0,image_path)
        rotate_image(90,image_path)
        rotate_image(180,image_path)
        rotate_image(270,image_path)
    print('finished annotations')


geometrically_transformed_files = [os.path.join(root, name)
            for root, dirs, files in os.walk(path_aug_images)
            for name in files
            if name.endswith(".jpg")]
geometrically_transformed_annotation_files = [os.path.join(root, name)
            for root, dirs, files in os.walk(path_aug_annotations)
            for name in files
            if name.endswith(".png")]

noise_brightness_aug=False
if noise_brightness_aug==True:
    
    i=0
    for image_path in geometrically_transformed_files:
        for multiply in [0.9,1.1]:
            for gauss in [0.04]:
                noise_brightness(i,multiply,gauss,image_path)
                noise_brightness(i,multiply,gauss,image_path)
                noise_brightness(i,multiply,gauss,image_path)
                noise_brightness(i,multiply,gauss,image_path)
                i+=1
    print('finished images')
    

    for image_path in geometrically_transformed_annotation_files:
        #print(image_path)
        multiply=1.1
        #print(path_aug_annotations + "/" + image_path.split("/")[-1].split(".png")[0] + "_gaussian_{}_mult_{}".format(gauss,multiply) + ".png")
        img = cv2.imread(image_path)
        for multiply in [0.9,1.1]:
            for gauss in [0.04]:
                cv2.imwrite(path_aug_annotations + "/" + image_path.split("/")[-1].split(".png")[0] + "_gaussian_{}_mult_{}".format(gauss,multiply) + ".png", img)
            cv2.imwrite(path_aug_annotations + "/" + image_path.split("/")[-1].split(".png")[0] + "_mult_{}".format(multiply) + ".png", img)
        cv2.imwrite(path_aug_annotations + "/" + image_path.split("/")[-1].split(".png")[0] + "_gaussian_{}".format(gauss) + ".png", img)
    print('finished annotations')



def create_txt(data_split,rotation):
    if data_split == 'train':
        # add names of augmented images to train and trainval txt files

        path = os.path.dirname(os.path.realpath(__file__))

        txt_path = path + "/ECU_SFA_SCH_HGR/ImageSets/ECU_SFA_SCH_HGR/"
        ann_path = path + "/ECU_SFA_SCH_HGR/SegmentationClass_ign/"

        # add names to txt files
        for txt_file_name in ['train']:#["train", "trainval"]:
            file1 = open(txt_path + '{}.txt'.format(txt_file_name), 'r') 
            Lines = file1.readlines()
            image_list_rot = []
            image_list_rot_mul = []
            image_list_rot_gauss = []
            image_list_rot_gauss_mul = []
            for image_name in Lines: # all annotation image names in folder
                image_name = image_name.strip()
                image_list_rot.append(image_name)
                image_list_rot_gauss.append(str(image_name + "_gaussian_0.04"))
                image_list_rot_mul.append(str(image_name + "_mult_0.9"))
                image_list_rot_mul.append(str(image_name + "_mult_1.1"))
                image_list_rot_gauss_mul.append(str(image_name + "_gaussian_0.04_mult_0.9"))
                image_list_rot_gauss_mul.append(str(image_name + "_gaussian_0.04_mult_1.1"))
                image_list_rot_gauss_mul.append(str(image_name + "_mult_0.9"))
                image_list_rot_gauss_mul.append(str(image_name + "_mult_1.1"))

                for angle in [90,180,270]:
                    image_list_rot.append(str(image_name + "_rot{}".format(angle)))
                    image_list_rot_gauss.append(str(image_name + "_rot{}".format(angle) + "_gaussian_0.04"))
                    for multiply in [0.9,1.1]:
                        image_list_rot_gauss_mul.append(str(image_name + "_rot{}".format(angle) + "_gaussian_0.04_mult_{}".format(multiply)))
                        image_list_rot_gauss_mul.append(str(image_name + "_rot{}".format(angle) + "_mult_{}".format(multiply)))
                        image_list_rot_mul.append(str(image_name + "_rot{}".format(angle) + "_mult_{}".format(multiply)))
                    
            image_list_rot_gauss = image_list_rot_gauss + image_list_rot # image_list_rot already contains all rotated and non-rotated images w/o further augmentation
            image_list_rot_mul = image_list_rot_mul + image_list_rot
            image_list_rot_gauss_mul = image_list_rot_gauss_mul  + image_list_rot_gauss
            print(len(image_list_rot_gauss))
            print(len(image_list_rot_mul))
            print(len(image_list_rot_gauss_mul))
            random.shuffle(image_list_rot)
            random.shuffle(image_list_rot_gauss)
            random.shuffle(image_list_rot_mul)
            random.shuffle(image_list_rot_gauss_mul)
            #with open(txt_path + '{}'.format(txt_file_name) + '_rot' + '.txt', 'w') as f:
                #for item in image_list_rot:
                    #f.write("%s\n" % item)
            with open(txt_path + '{}'.format(txt_file_name) + '_rot_gaussian_mult' + '.txt', 'w') as f:
                for item in image_list_rot_gauss_mul:
                    f.write("%s\n" % item)
            #with open(txt_path + '{}'.format(txt_file_name) + '_rot_gaussian' + '.txt', 'w') as f:
                #for item in image_list_rot_gauss:
                    #f.write("%s\n" % item)
            #with open(txt_path + '{}'.format(txt_file_name) + '_rot_mult' + '.txt', 'w') as f:
                #for item in image_list_rot_mul:
                    #f.write("%s\n" % item)
    elif data_split == 'val_test':
        # add names of augmented images to train and trainval txt files

        path = os.path.dirname(os.path.realpath(__file__))

        txt_path = path + "/ECU_SFA_SCH_HGR/ImageSets/ECU_SFA_SCH_HGR/"
        ann_path = path + "/ECU_SFA_SCH_HGR/SegmentationClass_ign/"

        # add names to txt files
        for txt_file_name in ['val','test']:#["train", "trainval"]:
            file1 = open(txt_path + '{}.txt'.format(txt_file_name), 'r') 
            Lines = file1.readlines()
            image_list_rot = []
            image_list_rot_dark = []
            image_list_rot_bright = []
            image_list_rot_gauss = []
            image_list_rot_gauss_dark = []
            image_list_rot_gauss_bright = []
            for image_name in Lines: # all annotation image names in folder
                image_name = image_name.strip()
                image_list_rot.append(image_name)
                image_list_rot_gauss.append(str(image_name + "_gaussian_0.04"))
                image_list_rot_dark.append(str(image_name + "_mult_0.9"))
                image_list_rot_bright.append(str(image_name + "_mult_1.1"))
                image_list_rot_gauss_dark.append(str(image_name + "_gaussian_0.04_mult_0.9"))
                image_list_rot_gauss_bright.append(str(image_name + "_gaussian_0.04_mult_1.1"))
                #image_list_rot_gauss_mul.append(str(image_name + "_mult_0.9"))
                #image_list_rot_gauss_mul.append(str(image_name + "_mult_1.1"))
                if rotation == True:
                    for angle in [90,180,270]:
                        image_list_rot.append(str(image_name + "_rot{}".format(angle)))
                        image_list_rot_gauss.append(str(image_name + "_rot{}".format(angle) + "_gaussian_0.04"))
                        image_list_rot_gauss_dark.append(str(image_name + "_rot{}".format(angle) + "_gaussian_0.04_mult_0.9"))
                        image_list_rot_gauss_bright.append(str(image_name + "_rot{}".format(angle) + "_gaussian_0.04_mult_1.1"))
                        #image_list_rot_gauss_mul.append(str(image_name + "_rot{}".format(angle) + "_mult_{}".format(multiply)))
                        image_list_rot_dark.append(str(image_name + "_rot{}".format(angle) + "_mult_0.9"))
                        image_list_rot_bright.append(str(image_name + "_rot{}".format(angle) + "_mult_1.1"))
                    
            #image_list_rot_gauss = image_list_rot_gauss + image_list_rot # image_list_rot already contains all rotated and non-rotated images w/o further augmentation
            #image_list_rot_mul = image_list_rot_mul + image_list_rot
            #image_list_rot_gauss_mul = image_list_rot_gauss_mul  + image_list_rot_gauss
            print(len(image_list_rot_gauss))
            print(len(image_list_rot_dark))
            print(len(image_list_rot_bright))
            print(len(image_list_rot_gauss_dark))
            print(len(image_list_rot_gauss_bright))
            #random.shuffle(image_list_rot)
            #random.shuffle(image_list_rot_gauss)
            #random.shuffle(image_list_rot_mul)
            #random.shuffle(image_list_rot_gauss_mul)
            if rotation == True:
                with open(txt_path + '{}'.format(txt_file_name) + '_rot' + '.txt', 'w') as f:
                    for item in image_list_rot:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_rot_gaussian_dark' + '.txt', 'w') as f:
                    for item in image_list_rot_gauss_dark:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_rot_gaussian_bright' + '.txt', 'w') as f:
                    for item in image_list_rot_gauss_bright:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_rot_gaussian' + '.txt', 'w') as f:
                    for item in image_list_rot_gauss:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_rot_dark' + '.txt', 'w') as f:
                    for item in image_list_rot_dark:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_rot_bright' + '.txt', 'w') as f:
                    for item in image_list_rot_bright:
                        f.write("%s\n" % item)
            elif rotation == False:
                with open(txt_path + '{}'.format(txt_file_name)+ '.txt', 'w') as f:
                    for item in image_list_rot:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_gaussian_dark' + '.txt', 'w') as f:
                    for item in image_list_rot_gauss_dark:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_gaussian_bright' + '.txt', 'w') as f:
                    for item in image_list_rot_gauss_bright:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_gaussian' + '.txt', 'w') as f:
                    for item in image_list_rot_gauss:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_dark' + '.txt', 'w') as f:
                    for item in image_list_rot_dark:
                        f.write("%s\n" % item)
                with open(txt_path + '{}'.format(txt_file_name) + '_bright' + '.txt', 'w') as f:
                    for item in image_list_rot_bright:
                        f.write("%s\n" % item)


create_txt('val_test', rotation=False)
