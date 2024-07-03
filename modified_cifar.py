"""
auth: AJ Boyd (aboyd3@umbc.edu)
date: 5/13/2024
desc: This is a script that uses TensorFlow to build a CNN (convulational neural network) model architecture,
trains that model on the CIFAR-10 image dataset, then evaluates its performance based on test data.
"""


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set a specific random seed for reproducibility
tf.random.set_seed(42)

# load in data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print(len(train_labels))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# added data augmentation
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# make the CNN model
# changed kernel size to 3x3
# added additional layers
model = models.Sequential()
model.add(layers.Conv2D(32,(3, 3), activation="relu", kernel_initializer="he_uniform", padding="same", input_shape=(32, 32, 3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.3))
model.add(layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_uniform", padding="same"))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.4))
# add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu", kernel_initializer="he_uniform"))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))

model.summary()  # display model architecture

# after model architecture is made, train the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

# use adam optimizer instead of sgd
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

steps = int(train_images.shape[0] / 64)
print("steps per epoch:", steps)
# changed # of epochs
# added validation data
history = model.fit(
    datagen.flow(train_images, train_labels, batch_size=64),
    # train_images,
    # train_labels,
    steps_per_epoch=steps,
    epochs=400,
    validation_data=(test_images, test_labels),
)

# test the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)

# accuracy per class
class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

# Prepare to count predictions for each class
correct_pred = {classname: 0 for classname in class_names}
total_pred = {classname: 0 for classname in class_names}

# Get model predictions on the test set
predictions = model.predict(test_images)
predicted_labels = tf.argmax(predictions, axis=1)

# Collect the correct predictions for each class
for true_label, predicted_label in zip(test_labels, predicted_labels):
    true_class = class_names[true_label[0]]
    if true_label == predicted_label:
        correct_pred[true_class] += 1
    total_pred[true_class] += 1

# Print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f"Accuracy for class: {classname:12s} is {accuracy:.1f} %")

# Print overall accuracy
overall_accuracy = 100 * (sum(correct_pred.values()) / sum(total_pred.values()))
print(f"Overall accuracy is {overall_accuracy:.1f} %")
