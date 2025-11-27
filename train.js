const natural = require("natural");
const fs = require("fs");

// Load training data
const trainingData = require("./data/training");

// Create classifier
const classifier = new natural.BayesClassifier();

// Normalize text
const normalize = (text) => text.toLowerCase().replace(/[^\w\s]/g, '');

// Add training data
trainingData.forEach(item => {
  classifier.addDocument(normalize(item.text), item.label);
});

// Train the classifier
console.log("⏳ Training classifier...");
classifier.train();

// Save trained model
classifier.save("sepitoriClassifier.json", (err) => {
  if (err) console.error("❌ Error saving classifier:", err);
  else console.log("✅ Classifier trained and saved as sepitoriClassifier.json");
});
