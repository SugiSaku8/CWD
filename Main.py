import json
import os
#↑標準ライブラリ
#↓非標準MLライブラリ
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.models import Sequential
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
embedding_dim =  16
vocab_size = len(tokenizer.word_index) +  1

model = Sequential([
    Embedding(vocab_size, embedding_dim),
    Dropout(0.2),
    GlobalAveragePooling1D(),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
print("モデルの設定が終了しました。")
# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("______________________")
print("モデルの学習を開始します")
# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
print("モデルの学習が終了しました。")
# Function to analyze malice in a question
def analyze_question(question):
    tokens = tokenizer.texts_to_sequences([question])
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=250)
    predictions = model.predict(padded_tokens)
    for word, prediction in zip(question.split(), predictions[0]):
        if prediction >  0.5:  # Threshold for maliciousness
            print(f"悪意のある言葉の可能性があります: {word}/スコア:${prediction}")
# Example usage
print("テストを行います。")
score = model.evaluate(X_test, y_test, verbose=0)
print(f"損失: {score[0]} / 精度: {score[1]}")
print("利用を開始します。")
analyze_question("おめぇは関係ない。黙れ。")
print("保存します。")
saved_model_path = '/app/Model/CWDmodel_V1'
model.save(saved_model_path, save_format='tf')
print(f"モデルは、 {saved_model_path}に保存されました。")