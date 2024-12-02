import nltk
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import random

# Carga de datos
with open('intents_spanish.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

lemmatizer = nltk.stem.WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '¿', '!', '¡', '.']

# Preparación de los datos
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Normalización y lematización
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(set(words))
classes = sorted(set(classes))

# Guardado de palabras y clases
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Creación del conjunto de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    bag = [1 if w in pattern_words else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([elem[0] for elem in training])
train_y = np.array([elem[1] for elem in training])

# Configuración del modelo
model = Sequential([
    Dense(256, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense( len(train_y[0]), activation='softmax')
])

# Optimizador y reducción de tasa de aprendizaje
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)

# Entrenamiento
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1, callbacks=[reduce_lr])

# Guardar el modelo
model.save('chatbot_model.h5')
print('Modelo creado y guardado exitosamente.')
