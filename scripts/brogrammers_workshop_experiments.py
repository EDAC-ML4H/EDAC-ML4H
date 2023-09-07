import os
import gc
import time
import numpy as np
import zipfile
import tempfile
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L1, L2, L1L2

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Load data set
def load_set(data_dir:str, set_name:str):
    set_path = os.path.join(data_dir, set_name + '.json')
    with open(set_path, 'r') as f:
        data = json.load(f)
    
    # Extract labels and MFCCs
    X = np.array(data['mfcc'])
    y = np.array(data['label'])
    
    X = X.reshape(X.shape[0], -1, 15, 1)
    
    # Encode labels
    y[y=='p'] = 1
    y[y=='n'] = 0
    y = y.astype(np.int32)
    
    del data
    
    return X, y

# Function to setup model
def setup_model():
    # There seems to be a bug when trying to prune with an activity regulariser.
    # The Tensor becomes out of scope for some reason
    model = Sequential()

    model.add(Conv2D(64, (3, 3), strides=(1, 1),padding='valid', input_shape=(302, 15, 1)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2), strides=(1, 1), padding='valid'))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))

    model.add(Flatten())

    model.add(Dense(256,
                    kernel_regularizer=L1L2(l1=3e-4, l2=4e-3),
                    bias_regularizer=L2(3e-3)))
    # ,activity_regularizer=L2(3e-4)
    model.add(Dropout(0.5))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(Dense(128,
                    kernel_regularizer=L1L2(l1=1e-3, l2=1e-2),
                    bias_regularizer=L2(1e-2)))
    # , activity_regularizer=L2(1e-3)
    model.add(Dropout(0.3))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(Dense(1, activation='sigmoid'))
    return model

# Function to evaluate model
def predict(model, X, exp_dir, exp_name, quantise=False, float16=False):    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                           tf.lite.OpsSet.SELECT_TF_OPS]
    
    if quantise:
        # 8-bit quantisation
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if float16:
        # 16-bit float quantisation
        converter.target_spec.supported_types = [tf.float16]
        
    quantized_tflite_model = converter.convert()
    
    # Check size difference
    model_file = os.path.join(exp_dir, exp_name + '.tflite')
    with open(model_file, 'wb') as f:
        f.write(quantized_tflite_model)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(model_file)
    
    size = os.path.getsize(zipped_file) / 10**6
    
    # Remove temp file
    os.remove(zipped_file)
    
    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    # Run individual inferences for average inference time calculation
    y_pred = []
    start_time = time.time()
    for i in range(X.shape[0]):
        input_data = X[i].reshape(input_shape).astype(input_dtype)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0]
        y_pred.append(prediction)
    
    inference_time = (time.time() - start_time)/X.shape[0]
    
    return np.array(y_pred), size, inference_time

# Function to evaulate 3 configuration for experiments
def evaluate(model, X, y, exp_dir):
    
    mkdir(exp_dir)
    
    results = dict()
    
    # Default evaulation
    exp_name = 'default'
    y_pred, size, inf_time = predict(model, X, exp_dir, exp_name, quantise=False, float16=False)
    auc = roc_auc_score(y, y_pred)
    
    results[exp_name] = {
        'size': size,
        'inf_time': inf_time,
        'auc': auc
    }
    
    # Int8 post training quantisation
    exp_name = 'int8'
    y_pred, size, inf_time = predict(model, X, exp_dir, exp_name, quantise=True, float16=False)
    auc = roc_auc_score(y, y_pred)
    
    results[exp_name] = {
        'size': size,
        'inf_time': inf_time,
        'auc': auc
    }
    
    # Float16 post training quantisation
    exp_name = 'float16'
    y_pred, size, inf_time = predict(model, X, exp_dir, exp_name, quantise=True, float16=True)
    auc = roc_auc_score(y, y_pred)
    
    results[exp_name] = {
        'size': size,
        'inf_time': inf_time,
        'auc': auc
    }
    
    return results

# Constant sparsity experiment setup
def run_exp(model_weights_path,
            learning_rate,
            loss,
            metrics,
            X_train, y_train,
            X_valid, y_valid,
            X_test, y_test,
            pruning_percentages,
            run,
            experiment='const',
            batch_size=32,
            epochs=10,
            verbose=0):
    
    experiment_names = {'const': 'Constant Sparsity',
                       'poly': 'Polynomial Decay'}
    
    curr_experiment = experiment_names[experiment]    
    
    # Dictionary to store results
    exp_results = dict()
    
    # Add baseline
    base_model = setup_model()
    base_model.load_weights(model_weights_path)
    base_model.compile(optimizer=Adam(learning_rate=learning_rate),
                       loss=loss,
                       metrics=metrics)
    
    print(f"Running {curr_experiment} experiment for base model (0% pruning).")
    # Calculate metrics for base model and map to 0% pruning sparsity
    exp_dir = os.path.join(run, '0.0')
    mkdir(exp_dir)
    exp_results['0.0'] = evaluate(base_model, X_test, y_test, exp_dir)
    
    for percent in pruning_percentages:
        print(f"Running {curr_experiment} experiment with {percent*100:.2f}% pruning.")

        if experiment == 'const':
            pruning_params = {'pruning_schedule':
                              tfmot.sparsity.keras.ConstantSparsity(percent, 
                                                                    0, 
                                                                    end_step=-1, 
                                                                    frequency=1)}
        elif experiment == 'poly':
            pruning_params = {'pruning_schedule':
                              tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0,
                                                                   final_sparsity=percent,
                                                                   begin_step=0,
                                                                   end_step=np.ceil(X_train.shape[0] / batch_size).astype(np.int32) * epochs)}
        
        # Creating pruning model
        model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(base_model, **pruning_params)
        model_for_pruning.compile(optimizer=Adam(learning_rate=learning_rate),
                                  loss=loss,
                                  metrics=metrics)

        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

        # Fine-tune pruned model
        model_for_pruning.fit(X_train, y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(X_valid, y_valid),
                              callbacks=callbacks,
                              verbose=verbose)

        # Apply mask
        model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

        # Calculate metrics and map to pruning sparsity
        exp_dir = os.path.join(run, str(percent))
        mkdir(exp_dir)
        exp_results[str(percent)] = evaluate(model_for_export, X_test, y_test, exp_dir)
    
    return exp_results

# Function to save results of experiment
def save_results(exp_results, name, run):
    with open(os.path.join(run, f'{name}_results.json'), 'w') as f:
        json.dump(exp_results, f)

if __name__ == '__main__':
    ####################################################################
    # Load datasets
    data_dir = '../coswara-brogrammers'
    X_train, y_train = load_set(data_dir, 'train')
    X_valid, y_valid = load_set(data_dir, 'valid')
    X_test, y_test = load_set(data_dir, 'test')
    
    ####################################################################
    # Load model weights
    model_path = './brogrammers.h5'

    ####################################################################
    # Evaluation metrics
    METRICS = ['accuracy']
    
    ####################################################################
    # Training parameters
    learning_rate = 0.0001
    # Use of variable for optimizer seems to be causing issues with reruns
    # Pass learning_rate instead and instantiate Adam as needed
    # optimizer = Adam(learning_rate=learning_rate)
    loss = 'binary_crossentropy'
    epochs = 10
    
    ####################################################################
    # Experiment parameters
    pruning_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999]
    
    # Repeted experiment run
    runs = 20
    base_seed = 1234
    experiment = 'const' #'poly'
    name = 'brogrammers_constant_sparsity' if experiment == 'const' else 'brogrammers_polynomial_decay'
    
    for run in range(runs):
        # Free up memory as much as possible before starting experiment
        tf.keras.backend.clear_session()
        gc.collect()
        
        tf.random.set_seed(base_seed + run)
        
        # Make directory to save results of run
        mkdir(str(run))
        
        print(f'Running experiment with random seed {base_seed + run}')
    
        ####################################################################
        # Run experiment
        exp_results = run_exp(model_path,
                              learning_rate,
                              loss,
                              METRICS,
                              X_train, y_train,
                              X_valid, y_valid,
                              X_test, y_test,
                              pruning_percentages,
                              str(run),
                              experiment=experiment,
                              epochs=epochs)

        # Save results of  experiment
        save_results(exp_results, name, str(run))
