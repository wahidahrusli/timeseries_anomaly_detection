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
2010-01-01  6.103308
2010-02-01  5.960629
2010-03-01  5.558788
2010-04-01  5.991353
2010-05-01  5.973774
               value
timestamp
2010-01-01  6.103308
2010-02-01  5.960629
2010-03-01  5.558788
2010-04-01  5.991353
2010-05-01  5.973774
```


-------

## Visualize the data

### Timeseries data without anomalies
Data for training: 

```python
fig, ax = plt.subplots()
gpp5_data_1.plot(legend=False, ax=ax)
plt.show()
```

![01_visualize_gpp5data1](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/01_visualize_gpp5data1.png)

### Timeseries data with anomalies
Data for testing: 

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
Number of training samples: 3987
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
Training input shape:  (3619, 368, 1)
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
    epochs=16,
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
26/26 [==============================] - 2s 90ms/step - loss: 0.7258 - val_loss: 0.3247
Epoch 2/16
26/26 [==============================] - 2s 68ms/step - loss: 0.2269 - val_loss: 0.1939
Epoch 3/16
26/26 [==============================] - 2s 82ms/step - loss: 0.1607 - val_loss: 0.1457
Epoch 4/16
26/26 [==============================] - 2s 76ms/step - loss: 0.1285 - val_loss: 0.1165
Epoch 5/16
26/26 [==============================] - 2s 69ms/step - loss: 0.1070 - val_loss: 0.0899
Epoch 6/16
26/26 [==============================] - 2s 71ms/step - loss: 0.0933 - val_loss: 0.0765
Epoch 7/16
26/26 [==============================] - 2s 69ms/step - loss: 0.0854 - val_loss: 0.0691
Epoch 8/16
26/26 [==============================] - 2s 85ms/step - loss: 0.0802 - val_loss: 0.0636
Epoch 9/16
26/26 [==============================] - 2s 76ms/step - loss: 0.0758 - val_loss: 0.0588
Epoch 10/16
26/26 [==============================] - 2s 71ms/step - loss: 0.0719 - val_loss: 0.0543
Epoch 11/16
26/26 [==============================] - 2s 72ms/step - loss: 0.0689 - val_loss: 0.0498
Epoch 12/16
26/26 [==============================] - 2s 85ms/step - loss: 0.0656 - val_loss: 0.0463
Epoch 13/16
26/26 [==============================] - 2s 70ms/step - loss: 0.0629 - val_loss: 0.0427
Epoch 14/16
26/26 [==============================] - 2s 83ms/step - loss: 0.0605 - val_loss: 0.0400
Epoch 15/16
26/26 [==============================] - 2s 84ms/step - loss: 0.0583 - val_loss: 0.0365
Epoch 16/16
26/26 [==============================] - 2s 78ms/step - loss: 0.0562 - val_loss: 0.0338
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
Reconstruction error threshold:  0.18792257822122685
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
Test input shape:  (3619, 368, 1)
```

![07_anomaly_samples](https://github.com/wahidahrusli/timeseries_anomaly_detection/blob/main/figures/07_anomaly_samples.png)

*Terminal:*
```
Number of anomaly samples:  550
Indices of anomaly samples:  (array([1708, 1716, 1723, 1724, 1737, 1739, 1740, 1741, 1744, 1745, 1747,
       1756, 1757, 1758, 1759, 1760, 1761, 1763, 1764, 1765, 1767, 1768,
       1769, 1772, 1773, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1783,
       1784, 1785, 1787, 1788, 1789, 1791, 1792, 1793, 1794, 1795, 1796,
       1797, 1799, 1800, 1801, 1803, 1804, 1805, 1806, 1807, 1808, 1809,
       1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819, 1820,
       1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831,
       1832, 1833, 1835, 1836, 1837, 1839, 1840, 1841, 1842, 1843, 1844,
       1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855,
       1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866,
       1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877,
       1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888,
       1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899,
       1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910,
       1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1921,
       1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932,
       1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943,
       1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954,
       1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965,
       1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976,
       1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987,
       1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998,
       1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
       2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,
       2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031,
       2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2041, 2042,
       2043, 2044, 2045, 2046, 2047, 2048, 2049, 2050, 2051, 2052, 2053,
       2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063, 2064,
       2065, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075,
       2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2085, 2086,
       2087, 2088, 2089, 2090, 2091, 2092, 2093, 2094, 2095, 2096, 2097,
       2098, 2099, 2100, 2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108,
       2109, 2110, 2111, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119,
       2120, 2121, 2122, 2123, 2124, 2125, 2126, 2127, 2128, 2129, 2130,
       2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141,
       2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152,
       2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163,
       2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174,
       2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185,
       2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196,
       2197, 2198, 2199, 2200, 2201, 2202, 2203, 2204, 2205, 2206, 2207,
       2208, 2209, 2210, 2211, 2212, 2213, 2214, 2215, 2216, 2217, 2218,
       2219, 2220, 2221, 2222, 2223, 2224, 2225, 2226, 2227, 2228, 2229,
       2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237, 2238, 2239, 2240,
       2241, 2242, 2243, 2244, 2245, 2246, 2247, 2248, 2249, 2250, 2251,
       2252, 2253, 2254, 2255, 2256, 2257, 2258, 2259, 2260, 2262, 2263,
       2264, 2265, 2266, 2267, 2270, 2271, 2274, 2275, 2282, 2283, 2290,
       2291, 2294, 2307, 2308, 2309, 2310, 2311, 2315, 2319, 2320, 2322,
       2323, 2327, 2328, 2330, 2331, 2332, 2333, 2334, 2335, 2336, 2337,
       2338, 2339, 2342, 2343, 2357, 2358, 2359, 2360, 2361, 2362, 2382],
      dtype=int64),)
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
