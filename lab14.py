import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# Config / hyperparams
VOCAB_SIZE = 20000    # keep top N words (num_words passed to load_data)
MAXLEN = 200
EMBED_DIM = 128
LSTM_UNITS = 64
BATCH_SIZE = 128
EPOCHS = 8
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def make_model(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, lstm_units=LSTM_UNITS, maxlen=MAXLEN):
    """
    Embedding -> SpatialDropout -> Bidirectional LSTM -> Dense -> output
    """
    inp = layers.Input(shape=(maxlen,), dtype="int32")
    # input_dim should match num_words used to load dataset (VOCAB_SIZE).
    # If you use reserved indices differently, you might need vocab_size + offset.
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=maxlen)(inp)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    return model


if __name__ == "__main__":
    # Load & prepare data
    print("Loading IMDB dataset...")
    try:
        (x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
    except Exception as e:
        raise RuntimeError("Failed to download/load IMDB dataset. Check network or dataset cache.") from e

    # pad sequences (post padding/truncating)
    x_train = keras.preprocessing.sequence.pad_sequences(
        x_train, maxlen=MAXLEN, padding="post", truncating="post"
    )
    x_test = keras.preprocessing.sequence.pad_sequences(
        x_test, maxlen=MAXLEN, padding="post", truncating="post"
    )

    # small validation split
    VAL_SPLIT = 0.1
    val_count = int(len(x_train) * VAL_SPLIT)
    x_val = x_train[:val_count]
    y_val = y_train[:val_count]
    x_train2 = x_train[val_count:]
    y_train2 = y_train[val_count:]

    print("Shapes:", x_train2.shape, x_val.shape, x_test.shape)

    # Build, compile
    model = make_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, "best_imdb_rnn.h5"),
            save_best_only=True, monitor="val_accuracy", mode="max"
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)
    ]

    # Train (shuffle explicitly)
    history = model.fit(
        x_train2, y_train2,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
        shuffle=True
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}   Test loss: {test_loss:.4f}")

    # save final model
    model.save(os.path.join(MODEL_DIR, "imdb_rnn_final.h5"))
    print("Saved model to", os.path.join(MODEL_DIR, "imdb_rnn_final.h5"))

    # safe sampling of test indices (ensure test set big enough)
    sample_count = 8
    n_test = len(x_test)
    sample_count = min(sample_count, n_test)
    idxs = np.random.choice(n_test, size=sample_count, replace=False)
    preds = (model.predict(x_test[idxs]) > 0.5).astype(int).reshape(-1)
    for i, idx in enumerate(idxs):
        print(f"Sample {i}: True={y_test[idx]} Pred={int(preds[i])}")
