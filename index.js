const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');

// Load the tokenizer state
const tokenizerState = JSON.parse(fs.readFileSync('tokenizer.json', 'utf8'));

// Recreate the tokenizer
const tokenizer = {
  wordIndex: tokenizerState.word_index,
  indexWord: tokenizerState.index_word,
  numWords: tokenizerState.num_words,
  documentCount: tokenizerState.document_count,
  textsToSequences: function(texts) {
    return texts.map(text => {
      // Split the text into words, remove special characters
      const words = text.split(/\W+/);
      // Map each word to its index and pad with zeros up to the desired sequence length
      const sequence = Array(this.numWords).fill(0);
      words.forEach(word => {
        const wordIndex = this.wordIndex[word];
        if (wordIndex !== undefined) {
          sequence[wordIndex] =  1;
        }
      });
      return sequence;
    });
  }
};

// Load the model
async function loadModel() {
  const model = await tf.loadLayersModel('./CWD_Model/model.json');
  return model;
}

// Predict function
async function predict(text) {
  const model = await loadModel();
  const sequences = tokenizer.textsToSequences([text]);
  const tensorInput = tf.tensor(sequences);
  const prediction = model.predict(tensorInput);
  return prediction;
}

// Example usage
predict("You are very loud and annoying.")
  .then(prediction => {
    console.log(prediction);
  })
  .catch(err => {
    console.error(err);
  });