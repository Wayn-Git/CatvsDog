# %%
# !git clone https://github.com/Wayn-Git/CatvsDog

# %%
# %cd CatvsDog

# %% [markdown]
# # Importing Libraries

# %%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# ## Assigning the Directory

# %%
train_dir = "../Data/train"
test_dir = "../Data/test"

# %% [markdown]
# ## Making a train and test generator 

# %%


trainGen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
testGen = ImageDataGenerator(rescale=1./255)


train_gen = trainGen.flow_from_directory(train_dir,
                                          target_size=(160,160), 
                                          batch_size=64,
                                            class_mode="binary")
test_gen = testGen.flow_from_directory(test_dir,
                                        batch_size=64 ,
                                        target_size=(160,160), 
                                        class_mode="binary")

# %% [markdown]
# ## Checking the number of samples 

# %%
print("training samples:", train_gen.samples)
print("test samples:", test_gen.samples)
print("class indices:", train_gen.class_indices)

# %%
## Initializing the model. Using Tranfer Learning for better training

# %%
TRANSFER_LEARNING = True

if TRANSFER_LEARNING:
    print("Creating Transfer Learning model with MobileNetV2...")

    base_model = MobileNetV2(
        input_shape=(160, 160, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False 

    model = models.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
else:
    print("Creating Lightweight Custom CNN...")

    model = models.Sequential([
        layers.Input(shape=(160,160,3)),

        Conv2D(16 , (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation = 'relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.25),

        GlobalAveragePooling2D(),

        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])


# %% [markdown]
# ## Compiling the Model

# %%
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# %% [markdown]
# ## Displaying the model summary

# %%
model.summary()

# %% [markdown]
# ## Early stopping to avoid over fitting and useless epochs

# %%
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# %% [markdown]
# ## Reduce Lr to make sure that the model learns (Slows down the model so it learns)

# %%
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',    
    factor=0.2,           
    patience=3,           
    min_lr=1e-7,           
    verbose=1             
)

# %% [markdown]
# ## Initizalizing a callbacks variable

# %%
callbacks = [early_stopping, reduce_lr]

# %% [markdown]
# ## Actual Training of the model

# %%
history = model.fit(train_gen,
                    epochs=30,
                    validation_data=test_gen,
                    verbose=1,
                    callbacks=callbacks,
                    steps_per_epoch=train_gen.samples // train_gen.batch_size,
                    validation_steps=test_gen.samples // test_gen.batch_size)

# %% [markdown]
# 

# %% [markdown]
# ## Displaying the Accuracy and Lose

# %%
test_loss, test_accuracy = model.evaluate(test_gen)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# %% [markdown]
# ## Visual Representation of the train accuracy, val accuracy, Train Lose and Val Lose (Val = Test)
# 

# %%
import matplotlib.pyplot as plt


plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# ### (Was used in collab)

# %%
# import os
# os.makedirs("model", exist_ok=True)

# %% [markdown]
# ### (Was used in collab)

# %%
# model.save("model/cat_dog_model.keras")

# %% [markdown]
# ## Predicting to see if it works

# %%
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = "Untitled.jpg"
img = image.load_img(img_path, target_size=(160, 160))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Predicted: Dog 🐶")
else:
    print("Predicted: Cat 🐱")



