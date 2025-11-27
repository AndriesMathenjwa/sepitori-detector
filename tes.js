const express = require("express");
const natural = require("natural");
const cors = require("cors");

const app = express();
app.use(express.json());
app.use(cors());

let classifier;

// Normalize text
const normalize = (text) => text.toLowerCase().replace(/[^\w\s]/g, "");

// Load trained classifier
natural.BayesClassifier.load("sepitoriClassifier.json", null, (err, loaded) => {
  if (err) console.error("❌ Failed to load model:", err);
  else {
    classifier = loaded;
    console.log("✅ Model loaded and ready!");
  }
});

// Prediction route
app.post("/predict", (req, res) => {
  if (!classifier)
    return res.status(503).json({ error: "Model not loaded yet" });

  const { text } = req.body;
  if (!text)
    return res.status(400).json({ error: "Text is required" });

  const lower = normalize(text);
  const words = lower.split(/\s+/);

  // ----- Get model words safely -----
  let modelWords = [];
  if (classifier.wordFrequencyCount) {
    modelWords = Object.keys(classifier.wordFrequencyCount);
  } else if (classifier.features) {
    modelWords = Object.keys(classifier.features);
  }

  // ----- Detect unseen words -----
  const unseenWords = words.filter(w => !modelWords.includes(w));

  // Classifier probabilities
  const probs = classifier.getClassifications(lower);
  const sepProb = probs.find(p => p.label === "sepitori")?.value || 0;
  const nonProb = probs.find(p => p.label === "non-sepitori")?.value || 0;

  // Normalize confidence
  const total = sepProb + nonProb || 1;
  const sepitoriConfidence = sepProb / total;
  const nonConfidence = nonProb / total;

  // ----- FINAL LABEL LOGIC -----
  let finalLabel = "not recognized";

  if (words.length > 0 && unseenWords.length === words.length) {
    // All words are unseen
    finalLabel = "not recognized";
  } else if (sepitoriConfidence >= 0.65 && sepProb > 0) {
    finalLabel = "sepitori";
  } else if (nonConfidence >= 0.65 && nonProb > 0) {
    finalLabel = "non-sepitori";
  } else {
    finalLabel = "mixed";
  }

  // Send response
  res.json({
    text,
    finalLabel,
    probabilities: {
      sepitori: sepProb,
      nonSepitori: nonProb,
      sepitoriConfidence,
      nonConfidence,
      unseenWords,
    },
  });
});

app.listen(5000, () => {
  console.log("🚀 API running on http://localhost:5000");
});
