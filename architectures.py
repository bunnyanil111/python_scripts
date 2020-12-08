import keras
import tensorflow as tf
input_shape = (224, 224, 3)

def MobileNetv1():


    input_layer = keras.layers.Input(shape=input_shape)
    model = keras.applications.mobilenet.MobileNet(input_shape=None, alpha=1.0, include_top=False,pooling = 'avg', weights='imagenet')(input_layer)
    out1=keras.layers.Dense(8,activation='relu',name='dense_01')(model)
    out2=keras.layers.Dense(4,activation='relu',name='dense_02')(out1)
    output=keras.layers.Dense(1,activation='sigmoid',name='dense_03')(out2)

    model = keras.Model(inputs=[input_layer], outputs=[output], name='mobilenet')

    return model

def Faceliveness():
    model = keras.models.load_model('faceliveness.h5')
    return model

def MobileNetv2():


    input_layer = keras.layers.Input(shape=input_shape)
    model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=None, alpha=1.0, include_top=False,pooling = 'avg', weights='imagenet')(input_layer)
    out1=keras.layers.Dense(8,activation='relu',name='dense_01')(model)
    out2=keras.layers.Dense(4,activation='relu',name='dense_02')(out1)
    output=keras.layers.Dense(1,activation='sigmoid',name='dense_03')(out2)

    model = keras.Model(inputs=[input_layer], outputs=[output], name='mobilenet')

    return model

def densenet121():


    input_layer = keras.layers.Input(shape=input_shape)
    model = keras.applications.densenet.DenseNet121(input_shape=None, include_top=False,pooling = 'avg', weights='imagenet')(input_layer)
    out1=keras.layers.Dense(8,activation='relu',name='dense_01')(model)
    out2=keras.layers.Dense(4,activation='relu',name='dense_02')(out1)
    output=keras.layers.Dense(1,activation='sigmoid',name='dense_03')(out2)

    model = keras.Model(inputs=[input_layer], outputs=[output], name='mobilenet')

    return model
def Densenet169():


    input_layer = keras.layers.Input(shape=input_shape)
    model = keras.applications.densenet.DenseNet169(input_shape=None, alpha=1.0, include_top=False,pooling = 'avg', weights='imagenet')(input_layer)
    out1=keras.layers.Dense(8,activation='relu',name='dense_01')(model)
    out2=keras.layers.Dense(4,activation='relu',name='dense_02')(out1)
    output=keras.layers.Dense(1,activation='sigmoid',name='dense_03')(out2)

    model = keras.Model(inputs=[input_layer], outputs=[output], name='mobilenet')

    return model

def Densenet201():


    input_layer = keras.layers.Input(shape=input_shape)
    model = keras.applications.densenet.DenseNet201(input_shape=None, alpha=1.0, include_top=False,pooling = 'avg', weights='imagenet')(input_layer)
    out1=keras.layers.Dense(8,activation='relu',name='dense_01')(model)
    out2=keras.layers.Dense(4,activation='relu',name='dense_02')(out1)
    output=keras.layers.Dense(1,activation='sigmoid',name='dense_03')(out2)

    model = keras.Model(inputs=[input_layer], outputs=[output], name='mobilenet')

    return model

def InceptionV3():


    input_layer = keras.layers.Input(shape=input_shape)
    model = keras.applications.inception_v3.InceptionV3(input_shape=None, alpha=1.0, include_top=False,pooling = 'avg', weights='imagenet')(input_layer)
    out1=keras.layers.Dense(8,activation='relu',name='dense_01')(model)
    out2=keras.layers.Dense(4,activation='relu',name='dense_02')(out1)
    output=keras.layers.Dense(1,activation='sigmoid',name='dense_03')(out2)

    model = keras.Model(inputs=[input_layer], outputs=[output], name='mobilenet')

    return model

def InceptionResnetV2():


    input_layer = keras.layers.Input(shape=input_shape)
    model = keras.applications.inception_resnet_v2.InceptionResNetV2(input_shape=None, alpha=1.0, include_top=False,pooling = 'avg', weights='imagenet')(input_layer)
    out1=keras.layers.Dense(8,activation='relu',name='dense_01')(model)
    out2=keras.layers.Dense(4,activation='relu',name='dense_02')(out1)
    output=keras.layers.Dense(1,activation='sigmoid',name='dense_03')(out2)

    model = keras.Model(inputs=[input_layer], outputs=[output], name='mobilenet')

    return model

def customCNN():
	# input_layer = keras.layers.Input(shape=input_shape)
	# def build_discriminator():
   """
   Create a discriminator network using the hyperparameter values defined below
   :return:
   """
   leakyrelu_alpha = 0.2
   momentum = 0.8
   input_shape = (224, 224, 3)
   input_layer = tf.keras.layers.Input(shape=input_shape)
   # Add the first convolution block
   dis1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_layer)
   dis1 = tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha)(dis1)
   # Add the 2nd convolution block
   dis2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(dis1)
   dis2 = tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha)(dis2)
   dis2 = tf.keras.layers.BatchNormalization(momentum=momentum)(dis2)
   # Add the third convolution block
   dis3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')(dis2)
   dis3 = tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha)(dis3)
   dis3 = tf.keras.layers.BatchNormalization(momentum=momentum)(dis3)
   # Add the fourth convolution block
   dis4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same')(dis3)
   dis4 = tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha)(dis4)
   dis4 = tf.keras.layers.BatchNormalization(momentum=0.8)(dis4)
   # Add the fifth convolution block
   dis5 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same')(dis4)
   dis5 = tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha)(dis5)
   dis5 = tf.keras.layers.BatchNormalization(momentum=momentum)(dis5)
   # Add the sixth convolution block
   dis6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')(dis5)
   dis6 = tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha)(dis6)
   dis6 = tf.keras.layers.BatchNormalization(momentum=momentum)(dis6)
   # Add the seventh convolution block
   dis7 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, padding='same')(dis6)
   dis7 = tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha)(dis7)
   dis7 = tf.keras.layers.BatchNormalization(momentum=momentum)(dis7)
   # Add the eight convolution block
   dis8 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=2, padding='same')(dis7)
   dis8 = tf.keras.layers.LeakyReLU(alpha=leakyrelu_alpha)(dis8)
   dis8 = tf.keras.layers.BatchNormalization(momentum=momentum)(dis8)
   dis8 = tf.keras.layers.Flatten()(dis8)
   # Add a dense layer
   dis9 = tf.keras.layers.Dense(units=32)(dis8)
   dis9 = tf.keras.layers.LeakyReLU(alpha=0.2)(dis9)
   output = tf.keras.layers.Dense(units=1024, activation='relu')(dis9)
   output = tf.keras.layers.Dense(units=8, activation='relu')(dis9)

   # Last dense layer - for classification
   output = tf.keras.layers.Dense(units=1, activation='sigmoid')(dis9)
   model = tf.keras.Model(inputs=[input_layer], outputs=[output], name='sampleCNNN')
   # print(model.summary())
   return model




if __name__ == "__main__":
    mobilenet= MobileNet()
    mobilenet.summary()

