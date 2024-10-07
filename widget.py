import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog
from ui_form import Ui_Widget
from itertools import combinations
from qiskit_ibm_provider import IBMProvider
# IBMProvider.save_account("c6114a56ebddc3cdae7337953a114c0d8bca94aa0e079ee893a7415b102dc3aab8b46db8306bd72726b9af72f42cad98f789cb15adcf1a466a99f0b79c124a43", overwrite="True")
# provider = IBMProvider()

# # Set the backend
# backend_name = 'ibm_kyoto'  # Example backend name
# backend = provider.get_backend(backend_name)

class Widget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)

        # Initialize variables
        self.csv_file = ""
        self.backend = "lightning.qubit"
        self.number_of_qubits = 5
        self.reps = 1
        self.dev = ""

        #IBMProvider.save_account("c6114a56ebddc3cdae7337953a114c0d8bca94aa0e079ee893a7415b102dc3aab8b46db8306bd72726b9af72f42cad98f789cb15adcf1a466a99f0b79c124a43", overwrite=True)
        #self.provider = IBMProvider()
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
        self.backend = text
        
        """
        if self.backend == 'Simulator':
            self.dev = qml.device("default.qubit", wires=self.number_of_qubits)
        else:
            #backend = self.provider.get_backend('ibm_kyoto')  # Example backend name
            self.dev = qml.device("qiskit.ibm", wires=self.number_of_qubits, backend=backend)
        """
        print(f'Backend selected: {self.backend}')

    def ZZFeatureMap(self,nqubits, data):
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
            print(reps)
            qml.RY(theta[reps * nqubits + i], wires=i)

    def qnn_circuit(self, inputs, theta):
        self.ZZFeatureMap(self.number_of_qubits, inputs)
        self.TwoLocal(nqubits=self.number_of_qubits, theta=theta, reps=self.reps)
        return qml.expval(qml.Hermitian(self.M_global, wires=[0]))

    def start_training(self):
        if not self.csv_file:
            QMessageBox.warning(self, "Warning", "Please select a CSV file before starting the training.")
            return
        
        try:
            epochs = int(self.ui.EpochsInput.toPlainText())
            train_test_split_ratio = int(self.ui.TrainSplitInput.toPlainText())
            batch_size = int(self.ui.BatchInput.toPlainText())
            self.reps = int(self.ui.RepsInput.toPlainText())
            print(self.reps)
            self.number_of_qubits = int(self.ui.NumOQubInput.toPlainText())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid numerical values.")
            return
        

        

        """
        if self.backend == 'Simulator':
            self.dev = qml.device("default.qubit", wires=self.number_of_qubits)
        else:
            self.dev = qml.device('qiskit.ibmq', wires=self.number_of_qubits, backend="ibm_kyoto")
        """

        print(f'Backend selected: {self.backend}')


        df = pd.read_csv(self.csv_file)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        x_tr, x_test, y_tr, y_test = train_test_split(X, y, train_size=train_test_split_ratio)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5)

        scaler = MaxAbsScaler()
        x_tr = scaler.fit_transform(x_tr)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)

        x_test = np.clip(x_test, 0, 1)
        x_val = np.clip(x_val, 0, 1)

        pca = PCA(n_components=self.number_of_qubits)
        xs_tr = pca.fit_transform(x_tr)
        xs_test = pca.transform(x_test)
        xs_val = pca.transform(x_val)

        
        print(xs_tr)

        state_0 = [[1], [0]]
        print(state_0 * np.conj(state_0).T)
        self.M_global = state_0 * np.conj(state_0).T

    
        self.dev = qml.device(self.backend, wires=self.number_of_qubits)
        qnn = qml.QNode(self.qnn_circuit, self.dev, interface="tf")

        #weights = {"theta": np.random.uniform(size=(self.number_of_qubits * 2))}
        weights = {"theta": 8}
        
        
        qlayer = qml.qnn.KerasLayer(qnn, weights, output_dim=1)

        
        model = tf.keras.models.Sequential([qlayer])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy()])

        #earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=1, restore_best_weights=True)

        print (xs_tr.shape)
        print (y_tr.shape)
        
        history = model.fit(xs_tr, y_tr, epochs=epochs, shuffle=True,
                            validation_data=(xs_val, y_val),
                            batch_size=1)                            

        
        model.save("qnn.h5")
        


        """
        
        # Load and preprocess data
        df = pd.read_csv(self.csv_file)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        x_tr, x_test, y_tr, y_test = train_test_split(X, y, train_size=train_test_split_ratio)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5)

        scaler = MaxAbsScaler()
        x_tr = scaler.fit_transform(x_tr)
        x_test = scaler.transform(x_test)
        x_val = scaler.transform(x_val)

        x_test = np.clip(x_test, 0, 1)
        x_val = np.clip(x_val, 0, 1)

        pca = PCA(n_components=self.number_of_qubits)
        xs_tr = pca.fit_transform(x_tr)
        xs_test = pca.transform(x_test)
        xs_val = pca.transform(x_val)

        # Build the QNN model
        qnn = qml.QNode(self.qnn_circuit, self.dev, interface="tf")
        weights = {"theta": np.random.uniform(size=(self.number_of_qubits * 2))}
        qlayer = qml.qnn.KerasLayer(qnn, weights, output_dim=1)

        model = tf.keras.models.Sequential([qlayer])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.BinaryAccuracy()])

        earlystop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=1, restore_best_weights=True)

        history = model.fit(xs_tr, y_tr, epochs=epochs, shuffle=True,
                            validation_data=(xs_val, y_val),
                            batch_size=batch_size,
                            callbacks=[earlystop])

        model.save("qnn.h5")
        QMessageBox.information(self, "Training Complete", "The model has been trained and saved as 'qnn.h5'.")

        # Optionally, plot the losses
        self.plot_losses(history)
        """

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
