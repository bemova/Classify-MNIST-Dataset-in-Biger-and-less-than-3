import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import argparse
import time


# read mnist dataset from tensorflow repository if it is not downloaded yet.
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# Weights
def init_weights(shape):
    """
    this method is responsible for initialize the weight with truncated normal distribution.
    oIf all of the weights are the same or zero, they will all have the same error and there is no source of asymmetry between the neurons.
    What we could do, instead, is to keep the weights very close to zero, and initialize them with truncated normal distribution.
    :param shape:
    :return:
    """
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)

# Bias
def init_bias(shape):
    """
    this method is responsible for initialize the biases with truncated normal distribution.
    :param shape: shape of the biases vector.
    :return: a tensorflow variable which is initialized with 0.1
    """
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)

# conv2xd
def conv2d(x, W):
    """
    this method is responsible for applying convolutional operation by using tensorflow.
    :param x: is [batch, high, width, channel]
    :param W: is [filter high, filter width, channels input, channels output]
    :return: the result_with_three_layers.txt of convolutional operation with strides of 2 * 2 and padding of SAME.
    """
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

# pooling
def max_pool_2by2(x):
    """
    this method is responsible for applying max pooling operator with strides of 2 * 2 and padding of SAME.
    :param x: is [batch, high, width, channel]
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# convolutional layer
def convolutional_layer(input_x, shape):
    """
    this method is responsible to calculate the convolution operation by using the tensorflow conv2d method.
    then we applied the relu to the convolutional operation result_with_three_layers.txt. reslu as an activation function is responsible to
    enable training model for deeper net, and it reduce the execution time in inference in compare to other activation functions.
    :param input_x: input vector
    :param shape: the shape of input in order to initialize weights and biases.
    :return: the result_with_three_layers.txt of convolutional operation after applying the relu actiovation function to the result_with_three_layers.txt.
    """
    W = init_weights(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x, W) + b)


#Blur filter
def apply_blur_filter(input_x, filter):
    """
    this method is responsible for appying the blure filter by using tensorflow convolutional operation.
    in this sample we get blur filter as an input, and then, we applied the convolutional operation with stride of 1 * 1
    and same padding in order to have blurred images.
    :param input_x: is [batch, high, width, channel]
    :param filter: a predefined blur filter.
    :return: the result_with_three_layers.txt of blurred image after applying the relu activation function on the result_with_three_layers.txt.
    """
    W = filter
    conv = tf.nn.conv2d(input_x, W, strides=[1, 1, 1, 1], padding='SAME')
    # the reson of using the activation function in CNN is to having nonlinearity int he neural network.
    # So, the network can learn more complex function.
    # without activation function the nn would be only able to learn function which is a linear combination of its input data.
    # the activation function that we used here is Relu.
    # relu or rectifier linear unit is calculated by this formula: f(x) = max(0, x);
    return tf.nn.relu(conv)


# fully connected layer
def normal_full_layer(input_layer, size):
    """
    the method of a fully connected layer. in this case also we have weight and bias initialization.
    :param input_layer: the faltted vector as an input
    :param size: the size or the number of neruns that we have in the fully connected layer. it is used for defining
    and initializing the weight and bias.
    :return: the result_with_three_layers.txt of vector multiplication of input vector to the wight plus bias.
    """
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    return tf.matmul(input_layer, W) + b


# Convert Labels
def convert_labels(ndarray):
    """
    this method is responsible for converting input vector for example from [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    to [1, 0]. in this method if the label is used for showing the handwriting number which is less than three, we convert it to [1, 0].
    otherwise, we convert it to [0, 1] which means the label is responsible for a handwritten number which is greater than or equal to three.
    :param ndarray: the labels numpy array for a batch size.
    :return: the converted label as a numpy array. in order to be able to have binary classification.
    """
    result = np.zeros((ndarray.shape[0], 2))
    for index, item in enumerate(ndarray, 0):
        if np.argmax(item) <= 2:
            result[index] = [1, 0]
        else:
            result[index] = [0, 1]
    return result


# Train
def train(conv_layers_count):
    """
    this is the the method for creating a tensorflow graph and define a session in order to be able to train the defined model.
    this method could be used to show what is the result_with_three_layers.txt of having one or two or three convolutional layers in our model
    and the effect of this desicion on the accuracy and the execution time for train and test for the relevant model.
    :param conv_layers_count: the number of convolutional layers which is needed to solve the problem.
    in this case conv_layers_count is one or two or three.
    :return: None
    """
    # start time of training
    start = time.time()
    # here we defined a blur filter with kernel size of 3 * 3
    blur_filter = np.zeros([3, 3, 1, 1])
    blur_filter[1, 1, :, :] = 0.25
    blur_filter[0, 1, :, :] = 0.125
    blur_filter[1, 0, :, :] = 0.125
    blur_filter[2, 1, :, :] = 0.125
    blur_filter[1, 2, :, :] = 0.125
    blur_filter[0, 0, :, :] = 0.0625
    blur_filter[0, 2, :, :] = 0.0625
    blur_filter[2, 0, :, :] = 0.0625
    blur_filter[2, 2, :, :] = 0.0625

    # a placeholder which will be feed in the session. this placeholder is used for input images for a batch size.
    # as you can see the size of input image for the batch size is 14 * 14 for satisfying the requirement of the test.
    x = tf.placeholder(tf.float32, shape=[None, 14 * 14])

    # this placeholder is used for corresponding labels for a batch size. this palceholder has undefined batch size which
    # will be defined at runtime and 2. 2 is responsible for the binary class that we have in our test.
    y_true = tf.placeholder(tf.float32, shape=[None, 2])

    # reshape the input image which is in size of 50 * 196 to 50 * 14 * 14 * 1. 50 is the batch size, and 1 is the
    # number of channel that we have. MNIST dataset is a grayscale dataset. therefore, we have images with one color channnel.
    x_image = tf.reshape(x, [-1, 14, 14, 1])


    # here we applyed the blur filter in the input images in order to satisfy the second requirment of the test.
    # indeed for appying the blur filter on the image, we applyed the convolutional operation on the images whit predefined weight of filter.
    blurred_images = apply_blur_filter(x_image, blur_filter)


    # the result_with_three_layers.txt of the blurred images is used as inputs for the first convolutional layer.
    # convo_1 is the first module of our CNN model with the kernel size of 3 * 3.
    # and the number of convolutional filter that we have in the first module is 16.
    convo_1 = convolutional_layer(blurred_images, shape=[3, 3, 1, 16])
    # the second module which is used in our CNN model is a max pooling module with kernel size of 2 * 2 and
    # the stride of 2 * 2. the padding that is used for maxpooling module is "SAME".
    # For the SAME padding, the output height and width are ceil(float(in_height) / float(2)) if the stride is 2 * 2
    convo_1_pooling = max_pool_2by2(convo_1)

    if conv_layers_count == 1:
        # we flat the result_with_three_layers.txt of first two modules which is consist of one Conv module and one max pool.
        # and used the flatted vector as an input for fully connected layer.
        flatted = tf.reshape(convo_1_pooling, [-1, 4 * 4 * 16])
    elif conv_layers_count == 2:
        # in this case we want to have two convolutional layers.
        # the result_with_three_layers.txt of the maxpool_1 is used as an input for the second convolution module.
        # convo_2 is the second Conv module with the kernel size of 3 * 3. the channel size for this module is the size of output channel from maxpool_1
        # which is the number of filters that we had in the first convolutional module(16).
        # and the number of convolutional filters that we have in the first module is 32.
        convo_2 = convolutional_layer(convo_1_pooling, shape=[3, 3, 16, 32])

        # we defined a second max pool after applying the second convolutional operation.
        # here we have a max pooling module with kernel size of 2 * 2 and
        # the stride of 2 * 2. the padding that is used for maxpooling module is "SAME".
        # convo_2_pooling = max_pool_2by2(convo_2)

        # flat the result_with_three_layers.txt of second layer in order to use the result_with_three_layers.txt as an input vector for the fully connected layer
        # in a case of having two conv layer.
        flatted = tf.reshape(convo_2, [-1, 2 * 2 * 32])
    elif conv_layers_count == 3:
        # same comment as line 172 to 176.
        convo_2 = convolutional_layer(convo_1_pooling, shape=[3, 3, 16, 32])

        # in this case we want to have three convolutional layers.
        # the result_with_three_layers.txt of the convo_2 is used as an input for the third convolution module.
        # convo_3 is the second Conv module with the kernel size of 3 * 3. the channel size for this module is the size of output channel from convo_2
        # which is the number of filters that we had in the second convolutional module(32).
        # and the number of convolutional filters that we have in the third module is 64.
        convo_3 = convolutional_layer(convo_2, shape=[3, 3, 32, 64])

        # flat the result_with_three_layers.txt of the third layer in order to use the result_with_three_layers.txt as an input vector for the fully connected layer.
        # in a case of having three conv layer.
        flatted = tf.reshape(convo_3, [-1, 1 * 1 * 64])


    # use the flatted vector as an input for the fully connected layer. in order to satisfy the requirment of the test,
    # our images are classified to two classes, therefore, the output of the fully connected layer is in size of two.
    # we applied the relu function as an activation function on the result_with_three_layers.txt as well.
    full_layer_one = tf.nn.relu(normal_full_layer(flatted, 2))

    # this is a placeholder in order to define the keep probability for the dropout module that we want to define.
    # dropout is a generalization technic for reducing overfitting in a neural network.
    # at each training step we randomly dropout set to nodes. in our solution we defined the keep probability 0.5 or 50%.
    hold_prob = tf.placeholder(tf.float32)
    full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)

    # y_pred or y hat is the result_with_three_layers.txt of our model for a batch size.
    y_pred = full_one_dropout

    # the cross entropy lost is used in the back propagation process in order to defined the loss
    # that we had corresponding to the predicted y(y_pred or y hat) in compare to the actual y or y_true.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

    # the optimizer which is used on the back propagation is Adam optimizer.
    # learning rate that we used in our model is 0.0001.
    # learning rate is determined how fast we want to update the weight during the optimization process.
    # if learning rate is too samll we faced to the slow training. and in a case of having too large learning rate,
    # the gradient not converge. for solving such dataset and this model we used 0.0001.
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    # our optimizer try to minimize the cross entropy lost in the back propagation.
    train = optimizer.minimize(cross_entropy)

    init = tf.global_variables_initializer()

    # the number of steps that we want to have in our training.
    steps = 3000
    # end of definign our model or defining the tensorflow graph.
    # in order to run the model or graph that we definde in the tenseflow, we have to define a session.
    with tf.Session() as sess:
        # initialize all variable node in the graph.
        sess.run(init)

        for i in range(steps):

            # get the 50 images as a batch with their relevant labels.
            # batch_x is images as a numpy array, and batch_y is the labels as numpy array.
            batch_x, batch_y = mnist.train.next_batch(50)

            # here we have to conver labels that we have in our batch of data because the labels are like:
            # 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], but in order to have binary classification
            # we have to convert labels to [1, 0] or [0, 1].
            # the method "convert_labels" is described in the place of declaration.
            batch_y = convert_labels(batch_y)

            # the first requirement of the problem is to downsample the original images to 14 x 14 pixels.
            # the original image size in MNIST dataset is a vector of 784(28 * 28).
            # first we have to reshape it to a tensor of 28 * 28 in order to be able to resize the image to 14 * 14.
            batch_tensor = tf.reshape(batch_x, [50, 28, 28, 1])
            # here we resize the 28 * 28 tensor to 14 * 14 tensor.
            batch_x = tf.image.resize_images(batch_tensor, [14, 14])
            # here we have to convert the define the image as a numpy array or vector of size 196.
            # therefore we reshape the tensor to 196, and by invoking the eval() method we can convert the the tensor to a numpy array.
            batch_x = tf.reshape(batch_x, [-1, 196]).eval()

            # run the session for calclating the tran node of the tensorflow graph.
            # and feed the placeholders that we defind in our graph.
            sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})

            # PRINT OUT A MESSAGE EVERY 100 STEPS
            if i % 100 == 0:
                print('Currently on step {}'.format(i))
                print('Accuracy is:')
                # Test the Train Model
                matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

                acc = tf.reduce_mean(tf.cast(matches, tf.float32))

                # in order to be able to test our model and calculate the accuracy. we need the test dataset.
                # but we have to change the test dataset as well. first we have to change the image size in the test set
                # from 28 * 28 to 14 * 14. second thing that we have to do is changing the labels to binary class.
                batch_x_test = mnist.test.images
                batch_y_test = mnist.test.labels

                batch_tensor_test = tf.reshape(batch_x_test, [-1, 28, 28, 1])
                batch_x_test = tf.image.resize_images(batch_tensor_test, [14, 14])
                batch_x_test = tf.reshape(batch_x_test, [-1, 196]).eval()

                batch_y_test = convert_labels(batch_y_test)

                # run the session in order to claculate the accuracy of the model for the test dataset.
                print(sess.run(acc, feed_dict={x: batch_x_test, y_true: batch_y_test, hold_prob: 1.0}))

    end = time.time()
    execution_time = end - start
    print("Execution time for {} convolutional layers is: {}".format(conv_layers_count, execution_time))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="classifies the MNIST dataset into two classes (less than 3 and equal or greater than 3")

    parser.add_argument("--con_layer_count", help="the number of convolutional layers which is needed", default=3, type=int, required=False)

    args = parser.parse_args()

    if args.con_layer_count == 1:
        print("################# Training is started for one convolutional layer #################")
        train(conv_layers_count=1)
    if args.con_layer_count == 2:
        print("################# Training is started for two convolutional layers #################")
        train(conv_layers_count=2)
    elif args.con_layer_count == 3:
        print("################# Training is started for three convolutional layers #################")
        train(conv_layers_count=3)
    else:
        print("the number of convolutional layers should be one or two or three")