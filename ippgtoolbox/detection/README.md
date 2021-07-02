# Package description

This package implements different approaches for region-of-interest (roi) detections and skin detections.
We distinguish roi from skin detection: - roi detection: Cutting a rough part from an image - skin detection: Fine-grained segmentation of skin
Usually there is a roi detection step before the skin detection. This is supposed to reduce the necessary computational effort for applying the more complex task of skin detection.

## Implemented roi detection approaches:

- _caffenet_: A neural network for face detection

## Implemented skin detection approaches:

- _cheref_: A thresholding approach combining the Ycbcr and HSV color spaces
  - Reference: Dahmani, Djamila; Cheref, Mehdi; Larabi, Slimane (2020): Zero-sum game theory model for segmenting skin regions. In: Image and Vision Computing 99, S. 103925. DOI: 10.1016/j.imavis.2020.103925.
- _deeplab_: A neural network based approach for segmentation tasks and with special training for skin segmentation
  - Reference for general deeplab approach: Chen, Liang-Chieh; Zhu, Yukun; Papandreou, George; Schroff, Florian; Adam, Hartwig (2018): Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. In: ECCV.
  - Reference for concrete deeplab approach: tba
- _deeplab_revised_: An implementation of the before mentioned deeplab approach in the keras framework to allow for a simpler integration into an existing processing pipeline
