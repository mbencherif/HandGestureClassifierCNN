import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense,\
    MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D,\
    Add, Multiply, Concatenate, Flatten,\
    LSTM
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import glorot_normal

from tensorflow.python.ops import rnn, rnn_cell


class FuseModel:
    def __init__(self, images_input, motions_input, output_size, batch_size=2, summarize_weights=False):
        self.image_input = images_input
        self.motions_input = motions_input
        self.output_size = output_size
        self.batch_size = batch_size
        self.single_frame_image = None
        self.single_frame_motion = None
        self.summarize_weights = summarize_weights
        self.prediction = self.prediction(images_input, motions_input)

    def prediction(self, images_input, motions_input):
        image_shape = list(images_input.shape[2:4])

        l2_reg = l2(10**(-9))
        default_init = glorot_normal(seed=None)

        images_input = tf.reshape(images_input, shape=[-1] + image_shape + [3])
        motions_input = tf.reshape(motions_input, shape=[-1] + image_shape + [2])
        self.single_frame_image = BaseImageModel(images_input)
        self.single_frame_motion = BaseMotionModel(motions_input)

        fused = tf.concat([self.single_frame_image.prediction, self.single_frame_motion.prediction],
                          axis=-1, name='image_motion_fuse')

        fused = Conv2D(filters=1024, kernel_size=1, strides=1, padding='valid', activation='relu',
                       kernel_initializer=default_init, use_bias=True, name='image_motion_conv_fuse')(fused)
        fused = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(fused)

        size = fused.shape[1] * fused.shape[2] * fused.shape[3]
        # flatten
        fused = tf.reshape(fused, shape=[self.batch_size, -1, size], name='flatten')

        lstm_input = Dense(2048, kernel_initializer=default_init, name='lstm_pre_process')(fused)

        # [batch, frames, 2048]
        lstm1 = lstm_input
        lstm_depths = [1024, 512]
        return_sequences = [True, False]
        for i, (depth, return_sequence) in enumerate(zip(lstm_depths, return_sequences)):
            lstm1 = LSTM(depth, stateful=True, return_sequences=return_sequence,
                         kernel_initializer=default_init,
                         name='lstm_{}'.format(i),
                         implementation=2)(lstm1)

        output = Dense(512, kernel_initializer=default_init, name="fc_last")(lstm1)

        logits = Dense(self.output_size, activation="linear", kernel_initializer=default_init,
                       name="output")(output)

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
        for i in range(5):
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
        for i in range(5):
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


class BaseMotionModel:
    def __init__(self, feature):
        self.input = feature
        self.prediction = self.prediction(feature)

    def prediction(self, motion_input):
        # motion frame processing
        # entry
        default_init = glorot_normal(seed=None)
        conv1 = Conv2D(filters=32, kernel_size=3, padding='same',
                       activity_regularizer=l2(0.0005),
                       kernel_initializer=default_init,
                       use_bias=False,
                       name='motion_entry')(motion_input)
        depths = [64, 128, 256, 512]
        pooling_layers = [True, True, True, True]
        activations = ['linear', 'linear', 'relu', 'relu']
        biases = [False, False, True, True]
        for pool, depth, activation, bias in zip(pooling_layers, depths, activations, biases):
            conv1 = Conv2D(filters=depth, kernel_size=5, padding='same',
                           use_bias=bias, activation=activation,
                           kernel_initializer=default_init,
                           activity_regularizer=l2(10**(-8)),
                           name='motion_conv_{}'.format(depth))(conv1)
            if pool:
                conv1 = AveragePooling2D(pool_size=2, strides=2)(conv1)
        conv1 = AveragePooling2D(pool_size=3, strides=(2, 2), padding='same')(conv1)

        return conv1