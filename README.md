# Timeseries anomaly detection using an Autoencoder
This script demonstrates how you can use a reconstruction convolutional autoencoder model to detect anomalies in timeseries data.

------
## Quick look at the data

```python
print(gpp5_data_1.head())
print(gpp5_data_2.head())
```


*Terminal:*
```
                 value
timestamp
2010-01-01  145.890191
2010-02-01  147.968363
2010-03-01  149.343890
2010-04-01  148.454258
2010-05-01  143.871115
                 value
timestamp
2010-01-01  168.658241
2010-02-01  173.070104
2010-03-01  144.474992
2010-04-01  147.505860
2010-05-01  180.401826
```


-------

## Visualize the data

### Timeseries data without anomalies
Data for testing: `GPP5-DHU-CG05-01 Regen Off Gas in CS & GPP5-DHU-CG05-02 Regen Off Gas in SS`

```python
fig, ax = plt.subplots()
gpp5_data_1.plot(legend=False, ax=ax)
plt.show()
```

![01_visualize_gpp5data1](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/01_visualize_gpp5data1.png)

### Timeseries data with anomalies
Data for testing: `GPP5-DHU-CG09-01 LPG Treater`

```python
fig, ax = plt.subplots()
gpp5_data_2.plot(legend=False, ax=ax)
plt.show()
```

![02_visualize_gpp5data2](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/02_visualize_gpp5data2.png)

------------

## Prepare training data
Get data values from the training timeseries data file and normalize the value data.

```python
# Normalize and save the mean and std we get,
# for normalizing test data.
training_mean = gpp5_data_1.mean()
training_std = gpp5_data_1.std()
df_training_value = (gpp5_data_1 - training_mean) / training_std
print("Number of training samples:", len(df_training_value))
```


*Terminal:*
```
Number of training samples: 3981
```

### Create sequence
Create sequences combining `TIME_STEPS` contiguous data values from the training data.

```python
TIME_STEPS = 368

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)

x_train = create_sequences(df_training_value.values)
print("Training input shape: ", x_train.shape)
```

*Terminal:*
```
Training input shape:  (3613, 368, 1)
```

------

## Build a model
We will build a convolutional reconstruction autoencoder model. The model will take input of shape (`batch_size`, `sequence_length`, `num_features`) and return output of the same shape. In this case, `sequence_length` is 368 and `num_features` is 1.

```python
model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()
```


*Terminal:*
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv1d (Conv1D)              (None, 184, 32)           256
_________________________________________________________________
dropout (Dropout)            (None, 184, 32)           0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 92, 16)            3600      
_________________________________________________________________
conv1d_transpose (Conv1DTran (None, 184, 16)           1808
_________________________________________________________________
dropout_1 (Dropout)          (None, 184, 16)           0
_________________________________________________________________
conv1d_transpose_1 (Conv1DTr (None, 368, 32)           3616
_________________________________________________________________
conv1d_transpose_2 (Conv1DTr (None, 368, 1)            225
=================================================================
Total params: 9,505
Trainable params: 9,505
Non-trainable params: 0
_________________________________________________________________
```

------
## Train the model
Please note that we are using x_train as both the input and the target since this is a reconstruction model.

```python
history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)
```

*Terminal:*
```
Epoch 1/16
26/26 [==============================] - 2s 87ms/step - loss: 0.4476 - val_loss: 0.2931       
Epoch 2/16
26/26 [==============================] - 2s 78ms/step - loss: 0.2006 - val_loss: 0.1303       
Epoch 3/16
26/26 [==============================] - 2s 80ms/step - loss: 0.1294 - val_loss: 0.0950       
Epoch 4/16
26/26 [==============================] - 2s 83ms/step - loss: 0.1007 - val_loss: 0.0718       
Epoch 5/16
26/26 [==============================] - 2s 85ms/step - loss: 0.0849 - val_loss: 0.0631       
Epoch 6/16
26/26 [==============================] - 3s 100ms/step - loss: 0.0727 - val_loss: 0.0621      
Epoch 7/16
26/26 [==============================] - 3s 113ms/step - loss: 0.0627 - val_loss: 0.0567      
Epoch 8/16
26/26 [==============================] - 2s 90ms/step - loss: 0.0544 - val_loss: 0.0619
Epoch 9/16
26/26 [==============================] - 2s 85ms/step - loss: 0.0482 - val_loss: 0.0567
Epoch 10/16
26/26 [==============================] - 2s 75ms/step - loss: 0.0432 - val_loss: 0.0527
Epoch 11/16
26/26 [==============================] - 2s 77ms/step - loss: 0.0396 - val_loss: 0.0469
Epoch 12/16
26/26 [==============================] - 2s 78ms/step - loss: 0.0364 - val_loss: 0.0468
Epoch 13/16
26/26 [==============================] - 2s 76ms/step - loss: 0.0339 - val_loss: 0.0492
Epoch 14/16
26/26 [==============================] - 2s 77ms/step - loss: 0.0320 - val_loss: 0.0541
Epoch 15/16
26/26 [==============================] - 2s 76ms/step - loss: 0.0305 - val_loss: 0.0587
Epoch 16/16
26/26 [==============================] - 2s 76ms/step - loss: 0.0289 - val_loss: 0.0508
```

Let's plot training and validation loss to see how the training went.

```python
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
```

![03_train_model](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/03_train_model.png)

-------

## Detecting anomalies
We will detect anomalies by determining how well our model can reconstruct the input data.

1. Find MAE loss on training samples.
2. Find max MAE loss value. This is the worst our model has performed trying to reconstruct a sample. We will make this the `threshold` for anomaly detection.
3. If the reconstruction loss for a sample is greater than this `threshold` value then we can infer that the model is seeing a pattern that it isn't familiar with. We will label this sample as an `anomaly`.

```python
# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)
```

![04_detect_anomalies](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/04_detect_anomalies.png)

*Terminal:*
```
Reconstruction error threshold:  0.1708456222271589
```

### Compare reconstruction
let's see how our model has recontructed the first sample. This is the 368 timesteps from year 1 of our training dataset.

```python
# Checking how the first sequence is learnt
plt.plot(x_train[0])
plt.plot(x_train_pred[0])
plt.show()
```

![05_compare_reconstruction](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/05_compare_reconstruction.png)

### Prepare test data

```python
def normalize_test(values, mean, std):
    values -= mean
    values /= std
    return values


df_test_value = (gpp5_data_2 - training_mean) / training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()

# Create sequences from test values.
x_test = create_sequences(df_test_value.values)
print("Test input shape: ", x_test.shape)

# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))
```

![06_prepare_test_data](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/06_prepare_test_data.png)

*Terminal:*
```
Test input shape:  (3613, 368, 1)
```

![07_anomaly_samples](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/07_anomaly_samples.png)

*Terminal:*
```
Number of anomaly samples:  3613
Indices of anomaly samples:  (array([   0,    1,    2, ..., 3610, 3611, 3612], dtype=int64),)
```

-------
## Plot anomalies

```python
# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
anomalous_data_indices = []
for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)

df_subset = gpp5_data_2.iloc[anomalous_data_indices]
fig, ax = plt.subplots()
gpp5_data_2.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()
```
![08_plot_anomaly](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/08_plot_anomaly.png)
