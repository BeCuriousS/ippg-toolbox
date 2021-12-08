# Implementations for skin detection

- **cheref**: threshold based approach
- **deeplab/deeplab_Xception65**: neural network for semantic segmentation specially trained for the task of skin detection; it uses the Xception65 backbone resulting in roughly 42M parameters
- **deeplab/deeplab_MobileNetV2**: neural network for semantic segmentation specially trained for the task of skin detection; it uses the MobileNetV2 backbone resulting in roughly 2M parameters

<!-- TODO add corresponding publication (cinc2021) -->

**If using this approach for your research please cite:**

> tba

## How to use

- For deeplab
  - you'll find the instructions in each subfolder
  - at the moment both implementations are meant to be run within a docker container where a folder with images is given to the container and then processed, i.e. each image is segmented
- For cheref: check the documentation

## Comparison of the implemented deeplab models

<table>
  <tr>
    <td align="center">Original frame</td>
    <td align="center" colspan="2">deeplab_Xception65</td>
    <td align="center" colspan="2">deeplab_MobileNetV2</td>
  </tr>
  <tr>
    <td align="center"></td>
    <td align="center">Segmented frame probability</td>
    <td align="center">Segmented frame mask ("-t 0.95")</td>
    <td align="center">Segmented frame probability</td>
    <td align="center">Segmented frame mask ("-t 0.95")</td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_0.jpg" border=3></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_0_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_0_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_0_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_0_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_1.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_1_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_1_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_1_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_1_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_2.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_2_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_2_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_2_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_2_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_3.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_3_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_3_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_3_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_3_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_4.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_4_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_4_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_4_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_4_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_5.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_5_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_5_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_5_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_5_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_6.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_6_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_6_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_6_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_6_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_7.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_7_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_7_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_7_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_7_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_8.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_8_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_8_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_8_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_8_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_9.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_9_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_9_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_9_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_9_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_10.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_10_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_10_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_10_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_10_segm.png"></img></td>
  </tr>
  <tr>
    <td align="center"><img src="./test/assets/test_run_deeplab_on_single_record/test_img_11.jpg"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_proba/test_img_11_segm.png"></img></td>
    <td align="center"><img src="./deeplab_Xception65/img/example_frames_segmented_mask/test_img_11_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_proba/test_img_11_segm.png"></img></td>
    <td align="center"><img src="./deeplab_MobileNetV2/img/example_frames_segmented_mask/test_img_11_segm.png"></img></td>
  </tr>
</table>
