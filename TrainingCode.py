# Import libraries
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Define a function to apply a low-pass filter to the sensor data
def low_pass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# Define a function to extract features from the sensor data
def extract_features(data, fs, N=10):
    # Apply a low-pass filter with a cutoff frequency of 10 Hz
    filtered_data = low_pass_filter(data, 10, fs, 5)
    
    # Find the peaks and valleys of the filtered data
    peaks, _ = find_peaks(filtered_data, distance=fs*0.6)
    valleys, _ = find_peaks(-filtered_data, distance=fs*0.6)
    
    # Find the highest peak and the lowest valley in each heartbeat cycle
    high_peaks = []
    low_valleys = []
    for i in range(len(peaks) - 1):
        high_peak = np.argmax(filtered_data[peaks[i]:peaks[i+1]]) + peaks[i]
        low_valley = np.argmin(filtered_data[peaks[i]:peaks[i+1]]) + peaks[i]
        high_peaks.append(high_peak)
        low_valleys.append(low_valley)
    
    # Take only the first N high peaks and low valleys, padding with zeros if necessary
    high_peaks = high_peaks[:N] + [0]*(N - len(high_peaks))
    low_valleys = low_valleys[:N] + [0]*(N - len(low_valleys))
    
    # Extract the x and y coordinates of the high peaks and low valleys as features
    x1 = np.array(high_peaks) / fs
    y1 = filtered_data[high_peaks]
    x2 = np.array(low_valleys) / fs
    y2 = filtered_data[low_valleys]
    features = np.stack([x1, y1, x2, y2], axis=1)
    # Return the features and the filtered data
    return features, filtered_data

# Load the train data set
train_data = np.load('data_train.npy')

# Separate the sensor data and the labels
X_train = train_data[:, :1000]
y_train = train_data[:, -2:]

# Define the sampling frequency fs = 100
fs = 100

# Define the number of high peaks and low valleys to use as features
N = 10

# Extract features from the train data set
X_train_features = []
for i in range(len(X_train)):
    features, filtered_data = extract_features(X_train[i], fs, N)
    X_train_features.append(features)

# Reshape the features to a 2D array
X_train_features = np.array(X_train_features).reshape(-1, 4*N)

# Train a Gradient Boosting Regressor for each label
model_S = GradientBoostingRegressor()
model_D = GradientBoostingRegressor()
model_S.fit(X_train_features, y_train[:, 0])
model_D.fit(X_train_features, y_train[:, 1])

# Save the trained models
joblib.dump(model_S, 'model_S.pkl')
joblib.dump(model_D, 'model_D.pkl')
