# get the CNN
base_model = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE,
                                                    include_top=False,      # no top
                                                    weights='imagenet')
base_model.trainable = False                                                # freeze it

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()             # get a new output layer. This will produce features 

prediction_layer = tf.keras.layers.Dense(len(class_names),activation='softmax') # new predictions layer

inputs = tf.keras.Input(shape=(224,224,3))

# set model for compilation
x = base_model(inputs, training=False)
outputs = global_average_layer(x)

# create transfer learning feature extractor
model_fe = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model_fe.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# specific method for getting frozen features
def get_bottleneck_features(model, input_imgs):
    """Retrieve features from a CNN configured to be used as a feature extractor"""
    features = model.predict(input_imgs, verbose=0)
    return features

# get one time features from frozen cnn
train_features = get_bottleneck_features(model_fe, train_dataset)
test_features = get_bottleneck_features(model_fe, test_dataset)


# create model top to train 

epoch_num = 50

input_shape = model_fe.output_shape[1]

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(input_shape,)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(prediction_layer)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

history = model.fit(x=train_features, y=train_labels,
                    validation_data=(test_features, test_labels),
                    batch_size=BATCH_SIZE,
                    epochs=epoch_num)
