import numpy as np
import keras
import cv2

class DataGeneratorSkillBorders(keras.utils.Sequence):  # Corrected class inheritance
    'Generates data for Keras'
    def __init__(self, df_labels, train=True, batch_size=32, dim=(128, 128), n_channels=3,
                 n_classes=10, shuffle=True, **kwargs):
        'Initialization'
        super().__init__(**kwargs)
        self.dim = dim
        self.train = train
        self.batch_size = batch_size
        self.df_labels = df_labels.sample(frac=0.8 if train else 0.2, axis=0)
        print(self.df_labels)
        self.n_channels = n_channels  # RGB or gray
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.df_labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate batch df view
        df_batch = self.df_labels.iloc[index * self.batch_size: (index + 1) * self.batch_size]

        # Generate data
        return self.__data_generation(df_batch)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # shuffle, insert indexes remain, position in df changes
        if self.shuffle:
            self.df_labels = self.df_labels.sample(frac=1).reset_index(drop=True)

    def __data_generation(self, df_batch):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, (_, row) in enumerate(df_batch.iterrows()):
            # Store sample
            X[i,] = self.load_frame(row.path, row.frame, self.dim[0], self.dim[1])
            # Store class
            y[i] = row.border

        return X, y

    def load_frame(self, path, frame_nr, dx, dy):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
        res, frame = cap.read()
        if not res:
            raise ValueError(f"Failed to read frame {frame_nr} from {path}")
        frame = cv2.resize(frame, (dx, dy))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Uncomment if necessary
        cap.release()
        return frame / 255.0