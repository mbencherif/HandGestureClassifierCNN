from tensorflow.python.keras.layers import Conv2D, Conv3D,\
    Dense,\
    MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D,\
    Add, Multiply, Concatenate, Flatten,\
    LSTM
from tensorflow.python.keras.applications import MobileNetV2
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.initializers import glorot_normal
import tensorflow as tf


from tensorflow.python.ops import rnn, rnn_cell


class FuseModel:
    def __init__(self, images_input, motions_input, output_size, batch_size=2, summarize_weights=False):
        self.image_input = images_input
        self.motions_input = motions_input
        self.output_size = output_size
        self.batch_size = batch_size
        self.summarize_weights = summarize_weights
        self.prediction = self.prediction(images_input, motions_input)

    def prediction(self, images_input, motions_input):
        image_shape = list(images_input.shape.as_list()[2:4])

        l2_reg = l2(10**(-9))
        default_init = glorot_normal(seed=None)

        image_frame = images_input
        image_frame = tf.reshape(image_frame, shape=[-1] + image_shape + [3], name='image_frame_collapse')
        base_model = MobileNetV2(input_shape=image_shape + [3],
                                 include_top=False,
                                 weights='imagenet')
        base_model.trainable = False
        base_model.summary()

        frame_output = base_model(image_frame)

        motion_frame = motions_input
        motion_frame = tf.reshape(motion_frame, shape=[-1] + image_shape + [2], name='motion_frame_collapse')

        filters = [32, 64, 128, 256, 512]
        poolings = [False, True, True, True, True]
        activations = ['linear', 'relu', 'relu', 'relu', 'relu']
        motion_conv = motion_frame
        for i, (num_filter, pooling, activation) in enumerate(zip(filters, poolings, activations)):
            if pooling:
                motion_conv = MaxPooling2D(strides=2)(motion_conv)
            motion_conv = Conv2D(filters=num_filter, kernel_size=3, padding='same', activation=activation,
                                 kernel_initializer=default_init, activity_regularizer=l2_reg,
                                 name='motion_conv_{}'.format(i))(motion_conv)
        motion_conv = AveragePooling2D(strides=2)(motion_conv)

        fused_frame = Concatenate()([frame_output, motion_conv])
        fused_frame = Conv2D(filters=512, kernel_size=1, kernel_initializer=default_init, activity_regularizer=l2_reg,
                             name="fused_frame_compress")(fused_frame)

        frames = tf.reshape(fused_frame, [self.batch_size, -1] + fused_frame.shape.as_list()[1:],
                            name='frame_expand')
        frames = Conv3D(filters=64, kernel_size=(7, 1, 1), padding='same',
                        kernel_initializer=default_init, activity_regularizer=l2_reg,
                        name='conv3d_1')(frames)
        frames = Conv3D(filters=32, kernel_size=3, padding='same',
                        kernel_initializer=default_init, activity_regularizer=l2_reg,
                        name='conv3d_2')(frames)

        size = frames.shape.as_list()
        size = size[2] * size[3] * size[4]
        flattened = tf.reshape(frames, [self.batch_size, -1, size], name='flatten')

        dense = Dense(256, activation='relu', kernel_initializer=default_init, activity_regularizer=l2_reg,
                      name='FC_1')(flattened)
        dense = LSTM(128, name='LSMT_1', stateful=False)(dense)
        logits = Dense(self.output_size, name='FC_final')(dense)

        return logits
