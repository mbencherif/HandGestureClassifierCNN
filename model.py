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
    def __init__(self, images_input, output_size, batch_size=2, summarize_weights=False):
        self.image_input = images_input
        self.output_size = output_size
        self.batch_size = batch_size
        self.summarize_weights = summarize_weights
        self.prediction = self.make_prediction(images_input)

    def make_prediction(self, images_input):
        image_shape = list(images_input.shape.as_list()[2:4])

        l2_reg = l2(10**(-9))
        default_init = glorot_normal(seed=None)

        image_frame = images_input
        image_frame = tf.reshape(image_frame, shape=[-1] + image_shape + [3], name='image_frame_collapse')
        image_conv = BaseImageModel(image_frame).prediction

        fused_frame = image_conv
        frames = tf.reshape(fused_frame, [self.batch_size, -1] + fused_frame.shape.as_list()[1:],
                            name='frame_expand')
        frames = Conv3D(filters=128, kernel_size=(7, 1, 1), padding='valid',
                        kernel_initializer=default_init, activity_regularizer=l2_reg,
                        name='conv3d_1')(frames)
        frames = Conv3D(filters=64, kernel_size=3, padding='valid',
                        kernel_initializer=default_init, activity_regularizer=l2_reg,
                        name='conv3d_2')(frames)

        size = frames.shape.as_list()
        size = size[2] * size[3] * size[4]
        flattened = tf.reshape(frames, [self.batch_size, -1, size], name='flatten')

        dense = Dense(256, activation='relu', kernel_initializer=default_init, activity_regularizer=l2_reg,
                      name='FC_1')(flattened)
        logits = Dense(self.output_size, name='FC_final')(dense)

        logits = tf.reduce_mean(logits, axis=1, name='average')
        return logits


class BaseImageModel:
    def __init__(self, feature):
        self.input = feature
        self.prediction = self.prediction(feature)

    def prediction(self, image_input):
        l2_reg = l2(10**(-9))
        default_init = glorot_normal(seed=None)

        def se_net(in_block, depth):
            x = GlobalAveragePooling2D()(in_block)
            x = Dense(depth // 16, activation='relu',
                      kernel_initializer=default_init,
                      bias_initializer='zeros')(x)
            x = Dense(depth, activation='sigmoid', kernel_regularizer=l2_reg)(x)
            return Multiply()([in_block, x])

        # single image frame processing
        # entry
        conv1 = Conv2D(32, kernel_size=3, strides=2,
                       padding="same", activation="relu",
                       kernel_initializer=default_init,
                       name="initial_3x3_conv_1")(image_input)
        conv1 = Conv2D(32, kernel_size=3, strides=1,
                       padding="same", activation="relu",
                       kernel_initializer=default_init,
                       name="initial_3x3_conv_2")(conv1)
        conv1 = Conv2D(32, kernel_size=3, strides=1,
                       padding="same", activation="relu",
                       activity_regularizer=l2_reg,
                       kernel_initializer=default_init,
                       name="initial_3x3_conv_3")(conv1)
        conv1_pool = MaxPooling2D(2, strides=2, padding='valid', name='stem_pool_1')(conv1)
        conv1_3 = Conv2D(32, kernel_size=3, strides=2, padding='same',
                         activation='relu', kernel_initializer=default_init,
                         name='conv1_reduced_1')(conv1)
        conv1 = Concatenate()([conv1_3, conv1_pool])
        conv1_3 = Conv2D(64, kernel_size=1, strides=1,
                         padding='same', activation='relu',
                         kernel_initializer=default_init,
                         name='stem_3x3_pre_conv')(conv1)
        conv1_3 = Conv2D(96, kernel_size=3, strides=1,
                         padding='same', activation='relu',
                         activity_regularizer=l2_reg,
                         kernel_initializer=default_init,
                         name='stem_3x3_conv')(conv1_3)
        conv1_7 = Conv2D(64, kernel_size=1, strides=1,
                         padding='same', activation='relu',
                         kernel_initializer=default_init,
                         name='stem_7x7_pre_conv')(conv1)
        conv1_7 = Conv2D(64, kernel_size=[7, 1], strides=1,
                         padding='same', activation='relu',
                         kernel_initializer=default_init,
                         name='stem_7x7_conv_factor_1')(conv1_7)
        conv1_7 = Conv2D(64, kernel_size=[1, 7], strides=1,
                         padding='same', activation='relu',
                         kernel_initializer=default_init,
                         name='stem_7x7_conv_factor_2')(conv1_7)
        conv1_7 = Conv2D(96, kernel_size=3, strides=1,
                         padding='same', activation='relu',
                         activity_regularizer=l2_reg,
                         kernel_initializer=default_init,
                         name='stem_7x7_post_conv')(conv1_7)
        conv1 = Concatenate()([conv1_3, conv1_7])
        conv1_pool = MaxPooling2D(2, strides=2, padding='valid', name='stem_pool_2')(conv1)
        conv1_3 = Conv2D(192, kernel_size=3, strides=2, padding='same',
                         activation='relu', kernel_initializer=default_init,
                         name='conv1_reduced_2')(conv1)
        conv1 = Concatenate()([conv1_3, conv1_pool])
        conv1 = se_net(conv1, depth=384)

        # middle flow
        # Inception-Resnet Block A
        depth = 384
        conv2 = conv1
        for i in range(3):
            conv1 = Conv2D(depth, kernel_size=1, strides=1, padding='valid',
                           activity_regularizer=l2_reg, kernel_initializer=default_init,
                           name='block_A_base_{}'.format(i))(conv1)

            conv2_1 = Conv2D(128, kernel_size=1, strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_A_1x1_conv_{}'.format(i))(conv2)

            conv2_3 = Conv2D(64, kernel_size=1, strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_A_3x3_pre_conv_{}'.format(i))(conv2)
            conv2_3 = Conv2D(128, kernel_size=3, strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_A_3x3_conv_{}'.format(i))(conv2_3)

            conv2_7 = Conv2D(32, kernel_size=1, strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_A_7x7_pre_conv_1_{}'.format(i))(conv2)
            conv2_7 = Conv2D(64, kernel_size=3, strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_A_7x7_pre_conv_2_{}'.format(i))(conv2_7)
            conv2_7 = Conv2D(128, kernel_size=3, strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_A_7x7_conv_{}'.format(i))(conv2_7)

            res_conv = Concatenate()([conv2_1, conv2_3, conv2_7])
            res_conv = Conv2D(depth, kernel_size=1, strides=1, padding='same',
                              activation='linear', kernel_initializer=default_init,
                              name='block_A_res_conv_projection_{}'.format(i))(res_conv)

            conv1 = Add(name="block_A_final_add_{}".format(i))([res_conv, conv1])
            conv1 = se_net(conv1, depth=depth)
            conv2 = conv1

        # Inception-Resnet Reduction A
        conv2_pool = MaxPooling2D(2, strides=2, padding='valid', name='red_A_pool_2')(conv2)
        conv2_3 = Conv2D(depth, kernel_size=3, strides=2, padding='same',
                         activation='relu', kernel_initializer=default_init,
                         name='red_A_conv_3')(conv2)
        conv2_7 = Conv2D(256, kernel_size=1, strides=1, padding='same',
                         activation='relu', kernel_initializer=default_init,
                         name='red_A_pre_conv_7')(conv2)
        conv2_7 = Conv2D(256, kernel_size=3, strides=1, padding='same',
                         activation='relu', kernel_initializer=default_init,
                         name='red_A_conv_7_factor_1')(conv2_7)
        conv2_7 = Conv2D(depth, kernel_size=3, strides=2, padding='same',
                         activation='relu', kernel_initializer=default_init,
                         name='red_A_conv_7_factor_2')(conv2_7)
        conv2 = Concatenate()([conv2_pool, conv2_3, conv2_7])

        # Inception-Resnet Block B
        depth = 1024
        conv1 = conv2
        for i in range(3):
            conv1 = Conv2D(depth, kernel_size=1, strides=1, padding='valid',
                           activity_regularizer=l2_reg, kernel_initializer=default_init,
                           name='block_B_base_{}'.format(i))(conv1)

            conv2_1 = Conv2D(192, kernel_size=1, strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_B_1x1_conv_{}'.format(i))(conv2)

            conv2_7 = Conv2D(128, kernel_size=1, strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_B_7x7_pre_conv_1_{}'.format(i))(conv2)
            conv2_7 = Conv2D(160, kernel_size=[1, 7], strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_B_7x7_pre_conv_2_{}'.format(i))(conv2_7)
            conv2_7 = Conv2D(192, kernel_size=[7, 1], strides=1, padding='same',
                             activation='relu', kernel_initializer=default_init,
                             name='block_B_7x7_conv_{}'.format(i))(conv2_7)

            res_conv = Concatenate()([conv2_1, conv2_7])
            res_conv = Conv2D(depth, kernel_size=1, strides=1, padding='same',
                              activation='linear', kernel_initializer=default_init,
                              name='block_B_res_conv_projection_{}'.format(i))(res_conv)

            conv1 = Add(name="block_B_final_add_{}".format(i))([res_conv, conv1])

            conv1 = se_net(conv1, depth=depth)
            conv2 = conv1

        # Inception-Resnet Reduction B
        conv2_pool = MaxPooling2D(3, strides=2, padding='same', name='red_B_pool_2')(conv2)
        conv2_3_1 = Conv2D(256, kernel_size=1, strides=1, padding='same',
                           activation='relu', kernel_initializer=default_init,
                           name='red_B_pre_conv_3_1')(conv2)
        conv2_3_1 = Conv2D(384, kernel_size=3, strides=2, padding='same',
                           activity_regularizer=l2_reg,
                           activation='relu', kernel_initializer=default_init,
                           name='red_B_conv_3_1')(conv2_3_1)
        conv2_3_2 = Conv2D(256, kernel_size=1, strides=1, padding='same',
                         activation='relu', kernel_initializer=default_init,
                           name='red_B_pre_conv_3_2')(conv2)
        conv2_3_2 = Conv2D(288, kernel_size=3, strides=2, padding='same',
                           activity_regularizer=l2_reg,
                           activation='relu', kernel_initializer=default_init,
                           name='red_B_conv_3_2')(conv2_3_2)
        conv2_7 = Conv2D(256, kernel_size=1, strides=1, padding='same',
                         activation='relu', kernel_initializer=default_init,
                         name='red_B_pre_conv_7')(conv2)
        conv2_7 = Conv2D(288, kernel_size=3, strides=1, padding='same',
                         activation='relu', kernel_initializer=default_init,
                         name='red_B_conv_7_factor_1')(conv2_7)
        conv2_7 = Conv2D(320, kernel_size=3, strides=2, padding='same',
                         activity_regularizer=l2_reg,
                         activation='relu', kernel_initializer=default_init,
                         name='red_B_conv_7_factor_2')(conv2_7)
        conv2 = Concatenate()([conv2_pool, conv2_3_1, conv2_3_2, conv2_7])

        # exit
        return conv2
