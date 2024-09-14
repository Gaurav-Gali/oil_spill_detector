import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Define image size and batch size
IMG_SIZE = 35
BATCH_SIZE = 2

class sar_DLM:
    # Prepare the dataset
    def __init__(self,img_path_input):
        self.img_path=img_path_input
        self.model_path='oil_spill_detector.h5'

    def preprocess(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            os.path.join('backend/api/detectionsModel/ImgDataSets','train'),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )

        val_generator = val_datagen.flow_from_directory(
            os.path.join('backend/api/detectionsModel/ImgDataSets','validation'),
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary'
        )
        self.trainCNN(train_generator,val_generator)

    # Define the CNN model
    def trainCNN(self,train_generator,val_generator):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        # Compile the model
        self.model.compile(
            optimizer=Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Train the model
        self.history=self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            epochs=2,
            validation_data=val_generator,
            validation_steps=val_generator.samples // BATCH_SIZE
        )
        validationAccuracy()
        save_Model()

    def validationAccuracy(self):
        # Plot training and validation accuracy
        plt.plot(self.history.history['accuracy'], label='accuracy')
        #plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.legend(loc='lower right')
        plt.show()
        print("Accuracy:",self.history.history['accuracy'][0])

    # Save the model
    def save_Model(self):
        self.model.save('oil_spill_detector.h5')
        self.model_path='oil_spill_detector.h5'


    def predict_oil_spill(self):
        # Load the trained model
        self.model = tf.keras.models.load_model(self.model_path)

        # Load and preprocess the image
        img = image.load_img(self.img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize

        # Predict
        predictions = self.model.predict(img_array)
        print(predictions)
        if predictions[0] > 0.95:
            return 1    #"Oil Spill Detected"
        else:
            return 0    #"No Oil Spill"

# Test the model on a new image
sd=sar_DLM('backend/api/TestSamples/img_test3.jpg')
res=sd.predict_oil_spill()
print(res)

'''print(predict_oil_spill('backend/api/TestSamples/img_test.jpg'))
print(predict_oil_spill('backend/api/TestSamples/img_test1.jpg'))
print(predict_oil_spill('backend/api/TestSamples/img_test2.jpg'))
print(predict_oil_spill('backend/api/TestSamples/img_test3.jpg'))
print(predict_oil_spill('backend/api/TestSamples/img_test4.jpg'))'''


