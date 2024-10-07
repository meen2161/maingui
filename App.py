import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
from sklearn.decomposition import PCA
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog
from ui_form import Ui_Widget
from itertools import combinations
from qiskit_ibm_provider import IBMProvider
from qnnlib import qnnlib


class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        # Initialize variables
        self.csv_file = ""
        self.backend = ""
        self.number_of_qubits = 0
        self.reps = 0
        self.dev = ""

        # Connect UI elements to methods
        self.ui.pushButton.clicked.connect(self.open_file_picker)
        self.ui.pushButton_2.clicked.connect(self.start_training)
        self.ui.comboBox.currentTextChanged.connect(self.update_backend)

    def open_file_picker(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if file_name:
            self.csv_file = file_name
            self.update_selected_file_label()

    def update_selected_file_label(self):
        self.ui.label_2.setText(f'Selected file: {self.csv_file}')

    def update_backend(self, text):
        if text == "Select a backend":
            self.backend = None
        else:
            self.backend = text
        print(f'Backend selected: {self.backend}')

    def ZZFeatureMap(self, nqubits, data):
        nload = min(len(data), nqubits)
        for i in range(nload):
            qml.Hadamard(i)
            qml.RZ(2.0 * data[:, i], wires=i)
        for pair in list(combinations(range(nload), 2)):
            q0 = pair[0]
            q1 = pair[1]
            qml.CZ(wires=[q0, q1])
            qml.RZ(2.0 * (np.pi - data[:, q0]) * (np.pi - data[:, q1]), wires=q1)
            qml.CZ(wires=[q0, q1])

    def TwoLocal(self, nqubits, theta, reps):
        for r in range(reps):
            for i in range(nqubits):
                qml.RY(theta[r * nqubits + i], wires=i)
            for i in range(nqubits - 1):
                qml.CNOT(wires=[i, i + 1])
        for i in range(nqubits):
            qml.RY(theta[reps * nqubits + i], wires=i)

    def qnn_circuit(self, inputs, theta):
        self.ZZFeatureMap(self.number_of_qubits, inputs)
        self.TwoLocal(nqubits=self.number_of_qubits, theta=theta, reps=self.reps)
        # expval = qml.expval(qml.Hermitian(self.M_global, wires=[0]))
        # print(expval)
        # return tf.cast(tf.math.real(expval), tf.float32)
        return qml.expval(qml.Hermitian(self.M_global, wires=[0]))

    def convert_complex_to_real(complex_tensor):
        return tf.math.real(complex_tensor)
    

    def start_training(self):
        if not self.csv_file:
            QMessageBox.warning(self, "Warning", "Please select a CSV file before starting the training.")
            return

        try:
            epochs = int(self.ui.EpochsInput.toPlainText())
            train_test_split_ratio = int(self.ui.TrainSplitInput.toPlainText())
            batch_size = int(self.ui.BatchInput.toPlainText())
            self.reps = int(self.ui.RepsInput.toPlainText())
            self.number_of_qubits = int(self.ui.NumOQubInput.toPlainText())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numerical values.")
            return
        # Set up the experiment with qnnlib
        self.run_qnnlib_experiment(
            data_path=self.csv_file,
            target_column='DIED',  # Assuming "Outcome" is the target column in your dataset
            test_size=1 - (train_test_split_ratio / 100),
            batch_size=batch_size,
            epochs=epochs,
            reps=self.reps
        )

    def run_qnnlib_experiment(self, data_path, target_column, test_size, batch_size, epochs, reps):
        qnn = qnnlib.qnnlib(nqubits=self.number_of_qubits, device_name=self.backend)

        # Set the output paths for model and progress
        model_output_path = 'qnn_model_output.h5'
        csv_output_path = 'training_progress.csv'
        loss_plot_file = 'loss_plot.png'
        accuracy_plot_file = 'accuracy_plot.png'
        print(f'Data path: {data_path}')
        print(f'Taget column: {target_column}')
        print(f'Test size: {test_size}')
        print(f'Batch size: {batch_size}')
        print(f'Epochs: {epochs}')
        print(f'Reps: {reps}')

        # Run the experiment using qnnlib
        qnn.run_experiment(
            data_path=data_path,
            target=target_column,
            test_size=test_size,
            model_output_path=model_output_path,
            csv_output_path=csv_output_path,
            loss_plot_file=loss_plot_file,
            accuracy_plot_file=accuracy_plot_file,
            batch_size=batch_size,
            epochs=epochs,
            reps=reps,
            scaler=MinMaxScaler(),
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            seed=1234
        )
        QMessageBox.information(self, "Training Complete", f"The model has been trained and saved as {model_output_path}.")

    def plot_losses(self, history):
        import matplotlib.pyplot as plt
        tr_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(tr_loss) + 1)

        plt.figure()
        plt.plot(epochs, tr_loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Widget()
    main_window.show()
    sys.exit(app.exec())
