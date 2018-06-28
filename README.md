In this repository, we trained a deep network to classify MNIST dataset into two classes, deciding whether a digit in the image is greater (or equal to) or less than three (3), i.e. digit >=3 or digit<3.


Some preprocessing should be performed on data first (all in Tensorflow and pytorch):

# Requirments: 

- Downsample the original images to 14 x 14 pixels
- Blurring the images

The network architecture must have 1-3 batch-normalized CNN layers (with 3 x 3 x N
Kernels, where N is an adjustable parameter) and 2x2 strides followed by a single fully
connected layer. Any other layers or components could be added if required. We would
like infer how many CNN layers (1, 2 or 3) results in the best performance of the network
based on the database we have. 
