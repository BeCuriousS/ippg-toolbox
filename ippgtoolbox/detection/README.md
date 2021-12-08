# Package description

This package implements different approaches for region-of-interest (roi) detections and skin detections.
We distinguish roi from skin detection:

- roi detection: Cutting a rough part from an image
- skin detection: Fine-grained segmentation of skin

Usually there is a roi detection step before the skin detection. This is supposed to reduce the necessary computational effort for applying the more complex task of skin detection. This also allows the skin detection to generate more accurate results.

## Implemented roi detection approaches:

- _caffenet_: A neural network for face detection.

## Implemented skin detection approaches:

- _cheref_: A thresholding approach combining the Ycbcr and HSV color spaces
  - Reference: Dahmani, Djamila; Cheref, Mehdi; Larabi, Slimane (2020): Zero-sum game theory model for segmenting skin regions. In: Image and Vision Computing 99, S. 103925. DOI: 10.1016/j.imavis.2020.103925.
- _deeplab_:

  - _deeplab_Xception65_: A neural network based approach for segmentation tasks and with special training for skin segmentation. Click [here](https://github.com/tensorflow/models/tree/master/research/deeplab) to check out the original implementation.

  - _deeplab_MobileNetV2_: An implementation of the before mentioned deeplab approach using the MobileNetV2 as backbone. Click [here](https://github.com/luyanger1799/Amazing-Semantic-Segmentation) to check out the original implementation.
