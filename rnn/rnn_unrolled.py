def dataset(full_seq_len =  200, timesteps = 25):
    x_train, y_train = [], []
    sin_wave = np.sin(np.arange(size))
    for step in range(sin_wave.shape[0]-timesteps):
        x_train.append(sin_wave[step:step+timesteps])
        y_train.append(sin_wave[step+timesteps])
    return np.array(x).reshape(len(y),timesteps,1),np.array(y).reshape(len(y),1)