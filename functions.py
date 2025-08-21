import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_history(history, model_name="ANN"):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Exactitud
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Entrenamiento")
    plt.plot(epochs_range, val_acc, label="Validación")
    plt.title(f"Exactitud - {model_name}")
    plt.xlabel("Épocas")
    plt.ylabel("Exactitud")
    plt.legend()

    # Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Entrenamiento")
    plt.plot(epochs_range, val_loss, label="Validación")
    plt.title(f"Pérdida - {model_name}")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.legend()

    plt.show()
    
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Exactitud en prueba: {test_acc*100:.2f}%")
    return test_acc

def confusion_and_errors(model, x_test, y_test, class_names, num_errors=10):
    y_pred = np.argmax(model.predict(x_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta Real")
    plt.show()

    # Ejemplos de predicciones erróneas
    errors = np.where(y_pred != y_true)[0]
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(errors[:num_errors]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[idx])
        plt.title(f"Pred: {class_names[y_pred[idx]]}\nReal: {class_names[y_true[idx]]}")
        plt.axis("off")
    plt.show()