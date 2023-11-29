import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np

def load_and_preprocess_data(df):
    X = df.drop('openfda.device_class', axis=1)
    y = df['openfda.device_class']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    return X_train_resampled, X_test, y_train_resampled, y_test

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    
    return model

def compile_and_train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=64):
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    
    optimizer = Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
    )
    
    return history

def evaluate_and_plot(model, X_test, y_test, history):
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(classification_report(y_test, y_pred))

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

# Load and preprocess data
X_train_resampled, X_test, y_train_resampled, y_test = load_and_preprocess_data(df_openFDA)

# Build model
input_shape = (X_train_resampled.shape[1],)
model = build_model(input_shape)

# Compile and train model
history = compile_and_train_model(model, X_train_resampled, y_train_resampled, X_test, y_test)

# Evaluate and plot results
evaluate_and_plot(model, X_test, y_test, history)
