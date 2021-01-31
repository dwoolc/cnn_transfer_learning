base_model = tf.keras.applications.EfficientNetB0(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')


trainable_cnn = cnn_block_unfreeze(block_nums_to_train, base_model, model_name)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

checkpoint_dir = '/content/drive/My Drive/model_checkpoints/testmodel_'+ str(epoch_num_restore_phase1c)

classifer_top = tf.keras.models.load_model(checkpoint_dir)

#Compile phase 2 model
inputs = tf.keras.Input(shape=(224,224,3))
x = trainable_cnn(inputs, training=False)
x = global_average_layer(x)
outputs = classifer_top(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001

previous_epoch_num = 0

model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=previous_epoch_num, # set to 0 if using base model
                         # else is epoch num cnn retrieved from
                         validation_data=test_datasets)
