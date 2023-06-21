### NEURAL NETWORK TRAINING
# import necessary libraries
import matplotlib
matplotlib.use("Qt5Agg", force = True)
import matplotlib.pyplot as plt

import tensorflow as tf

from keras import regularizers

# prepare merged dataframe
exec(open('prep_df_7744044.py').read())

# create regressor dictionary for model implementation
regr_dct = {"vanilla_net": merged_df.drop(columns=["date", "index", "nonfarm", "rec"]),
            "macro_net": merged_df.drop(columns=["date", "spread_y", "rec"])}

for k, v in regr_dct.items():
    # split dataset into training set for network training
    X_train = v[:275].values
    y_train = merged_df["rec"][:275].values

    # set up neural network for prediction
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(len(v.columns),)))
    model.add(tf.keras.layers.Dense(units=9, activation="relu"))
    model.add(tf.keras.layers.Dense(units=5, activation="relu", kernel_regularizer=regularizers.l2(.01)))
    model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(learning_rate=.002)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.Precision(thresholds=.5)])

    # fit to training set
    hist = model.fit(X_train, y_train, batch_size=275, epochs=10000, verbose=0)

    # save model to given path
    model.save(filepath=k)

    # plot the loss as a function of the amount of epochs
    loss_vals = hist.history["loss"]
    epochs = range(1, len(loss_vals) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    plt.plot(epochs, loss_vals, label="Training loss")
    plt.title("Binary cross entropy loss as function of epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()