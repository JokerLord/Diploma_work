import numpy as np
from pathlib import Path
from tensorflow.keras import models, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

FOLDER = 'D:\\Learning\\MSU\\Year 4th\\Diploma Files'


class Dataset:

    def __init__(self, path: str):
        self.is_loaded = False
        p = Path(FOLDER) / Path(path)
        if p.exists():
            # print(f'Loading dataset from {path}')
            np_obj = np.load(str(p))
            self.images = np_obj['data']
            self.labels = np_obj['labels']
            self.gt_vectors = np_obj['gt_vectors']
            self.n_files = self.images.shape[0]
            self.is_loaded = True
            # print(f'Done. Dataset consists of {self.n_files} images.')

    def image(self, i):
        # read i-th image in dataset and return it as numpy array
        if self.is_loaded:
            return self.images[i, :, :, :]

    def images_seq(self, n=None):
        # sequential access to images inside dataset
        for i in range(self.n_files if not n else n):
            yield self.image(i)

    def batches_seq(self, batch_size=32):
        num = self.n_files // batch_size
        for i in range(num):
            yield self.images[i * batch_size: (i + 1) * batch_size, :, :, :]
        if self.n_files % batch_size:
            yield self.images[num * batch_size:, :, :, :]

    def random_image_with_label_and_gt_vector(self):
        # get random image with label from dataset
        i = np.random.randint(self.n_files)
        return self.image(i), self.labels[i], self.gt_vectors[i]

    def random_batch_with_gt_vectors(self, n):
        # create random batch of images with labels (is needed for training)
        indices = np.random.choice(self.n_files, n)
        imgs = []
        for i in indices:
            img = self.image(i)
            imgs.append(self.image(i))
        logits = np.array([self.gt_vectors[i] for i in indices])
        return np.stack(imgs), logits

    def image_with_label(self, i: int):
        # return i-th image with label from dataset
        return self.image(i), self.labels[i]


class MySimpleClassifier:

    def __init__(self, input_shape, n_output_channels, n_filters):
        self.model = models.Sequential()
        self.model.add(Input(shape=input_shape))
        self.model.add(Conv2D(n_filters, 3, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(n_filters, 3, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(n_filters * 2, 3, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(n_filters * 2, 3, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(n_filters * 4, 3, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(n_filters * 4, 3, activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(GlobalAveragePooling2D())

        self.model.add(Flatten())
        self.model.add(Dense(n_output_channels, activation='softmax'))

    def save(self, name: str):
        p = Path(name) / Path('my_checkpoint')
        self.model.save_weights(str(p))

    def load(self, name: str):
        p = Path(name) / Path('my_checkpoint')
        self.model.load_weights(str(p))

    def train(self, dataset: Dataset, val_images, val_gt_vectors):
        print(f'training started')
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss=CategoricalCrossentropy(), metrics=['accuracy'])
        self.model.fit(dataset.images,
                       dataset.gt_vectors,
                       epochs=7,
                       validation_data=(val_images, val_gt_vectors))
        print(f'training done')

    def test_on_dataset(self, dataset: Dataset):
        predictions = []
        for batch in dataset.batches_seq():
            predictions.extend(self.test_on_batch(batch))
        return predictions

    def test_on_batch(self, batch: np.ndarray):
        prediction = np.argmax(self.model.predict(batch), axis=1)
        return prediction
