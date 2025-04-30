import os
import numpy as np
import cnn
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    result_path = "model_results"

    X_train, X_val, y_train, y_val = cnn.load_audio()

    input_shape = (X_train.shape[1], X_train.shape[2], 1)  # (Time Frames, Mel Bins, Channels)
    num_classes = cnn.NUM_CLASSES

    model = cnn.create_model(input_shape, num_classes)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()


    # Compute class weights
    y_labels = np.argmax(y_train, axis=1)  # Convert one-hot back to lables
    class_weights = compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    class_weights_dict[1] += 0.2
    class_weights_dict[2] += 0.2
    print("Class Weights:", class_weights_dict)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        verbose=2,
        class_weight=class_weights_dict
    )

    model.save(os.path.join(result_path, "model.h5"), include_optimizer=False)
    
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")

    plt.plot(history.history["accuracy"])
    plt.plot(history.history['val_accuracy'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Model Accuracy and Loss")
    plt.ylabel("Value")
    plt.xlabel("Epoch")
    plt.legend(["Train Accuracy", "Validation Accuracy", "Train Loss", "Validation Loss"])
    plt.savefig(os.path.join(result_path, "model_results.png"))
    # plt.show()

    # Get model predictions on the test set
    y_pred = model.predict(X_val, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert from one-hot encoding to class labels
    y_true = np.argmax(y_val, axis=1)  # Convert true labels from one-hot to class labels

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Normalize the confusion matrix by dividing each row by the sum of that row to get percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=cnn.emotion_to_number.keys(), yticklabels=cnn.emotion_to_number.keys())    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(result_path, "confusion_matrix.png"))
    # plt.show()
