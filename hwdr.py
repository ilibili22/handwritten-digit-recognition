# Handwritten Digit Recognition: 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def main():

    x_train, y_train, x_test, y_test = run_data_set()

    x_train_resized, x_test_resized = image_resizer(x_train, x_test)
    # print(np.shape(x_train_resized))

    x_train_normalized, x_test_normalized = normalization(x_train_resized, x_test_resized)

    y_train_hot, y_test_hot = one_hot_doer(y_train, y_test)

    # image_opener(x_train_resized, y_train)

    model_fit_history = creating_model(x_train_normalized, x_test_normalized, y_train_hot, y_test_hot)

    show(model_fit_history)


def run_data_set():
    
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    return x_train, y_train, x_test, y_test



def image_resizer(x_train, x_test):
    x = x_train
    return tf.image.resize(x[..., tf.newaxis], (28,28)) , tf.image.resize(x_test[..., tf.newaxis], (28,28))



def normalization(x_train, x_test):

    normal_x_train = x_train / 255
    normal_x_test = x_test / 255

    return normal_x_train, normal_x_test
    


def one_hot_doer(y_train, y_test):

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return y_train, y_test




def image_opener(resized_img, y_train):

    title = 2
    y = y_train[title]

    plt.imshow(resized_img[title][:,:,0], cmap = "gray")
    plt.title(y)

    plt.show()


# write it again to understand more:
def creating_model(x_train, x_test, y_train, y_test):

    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)

    model = tf.keras.Sequential([

        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3,3),
            activation= "relu", 
            padding="same"
        ),
        # test it:
        tf.keras.layers.MaxPooling2D(pool_size =(2,2)),

        tf.keras.layers.Conv2D(32,(3,3), activation="relu", padding="same"),

        tf.keras.layers.MaxPooling2D(pool_size =(2,2)),

        tf.keras.layers.Conv2D(32,(3,3), activation="relu", padding="same"),
        
        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax")
    # ChatGPT:
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ChatGPT:
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1,
        random_state=42
    )


    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data =(x_val, y_val) )

    # ChatGPT:
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print("Test Accuracy:", round(test_accuracy,2))
    print("Test Loss:", round(test_loss,2))

    return history



# ChatGPT:

def show(history):

    # Accuracy Plot
    plt.figure(figsize=(10, 5))

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.show()


    # Loss Plot
    plt.figure(figsize=(10, 5))

    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")

    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.show()





if __name__ == "__main__":
    main()