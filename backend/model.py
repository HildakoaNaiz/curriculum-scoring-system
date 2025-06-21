import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class CurriculumScorer:
    def __init__(self):
        self.max_len = 100
        self.tokenizer = Tokenizer(num_words=5000)
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv1D(32, 3, activation='relu', input_shape=(self.max_len, 1)))
        model.add(MaxPooling1D(2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def preprocess(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post')
        return np.expand_dims(padded, axis=2)

    def score(self, text):
        processed = self.preprocess([text])
        score = self.model.predict(processed)[0][0]
        return float(score)

    def explain(self, text):
        # Placeholder explanation
        return "The score is based on keyword presence and structure."
