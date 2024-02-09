import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import UnitNorm
from sklearn.model_selection import train_test_split

print("すべてのパッケージのインポートが完了しました。")

# Load JSON data
with open('data.json','r') as f:
    data = json.load(f)
print("データセットのインポートが完了しました。")

# Preprocess data
sentences = [item['text'] for item in data]
labels = [item['label'] for item in data]
print("データセットの分割が完了しました。")

# Tokenize sentences
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
print("トークナイザーノ処理が完了しました。")

# Prepare data for training
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=250)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=250)
y_train = np.array(y_train)
y_test = np.array(y_test)
print("データの順部が完了しました。")

# Define the model
embedding_dim =   16
vocab_size = len(tokenizer.word_index) +   1

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    Dropout(0.2),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01), kernel_constraint=UnitNorm())
])

print("モデルの設定が終了しました。")

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("______________________")
print("モデルの学習を開始します")

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[early_stopping])

print("モデルの学習が終了しました。")

# Function to analyze malice in a question
def analyze_question(question):
    tokens = tokenizer.texts_to_sequences([question])
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=250)
    predictions = model.predict(padded_tokens)
    for word, prediction in zip(question.split(), predictions[0]):
        if prediction >   0.5:  # Threshold for maliciousness
            print(f"悪意のある言葉の可能性があります: {word}/スコア:${prediction}")

# Example usage
print("テストを行います。")
score = model.evaluate(X_test, y_test, verbose=0)
print(f"損失: {score[0]} / 精度: {score[1]}")
print("利用を開始します。")

#実践編
question = "おまえ、天才だなー"
tokens = tokenizer.texts_to_sequences([question])
padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=250)
predictions = model.predict(padded_tokens)
for word, prediction in zip(question.split(), predictions[0]):
    if prediction >   0.5:  # Threshold for maliciousness
        print(f"悪意のある言葉の可能性があります: {word}/スコア:${prediction}")

#関数を利用
print("_________________________________________")
analyze_question("You are very loud and annoying.")
print("保存します。")
saved_model_path = './Model/CWDmodel_V1'
model.save(saved_model_path, save_format='tf')
print(f"モデルは、 {saved_model_path}に保存されました。")
