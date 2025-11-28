const express = require("express");
const natural = require("natural");
const cors = require("cors");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(express.json());
app.use(cors());

let classifier;
const stemmer = natural.PorterStemmer;

// Normalize text: lowercase + remove punctuation
const normalize = (text) => text.toLowerCase().replace(/[^\w\s]/g, "");

// Stem words
const stemWords = (words) => words.map((w) => stemmer.stem(w));

// Stopwords to ignore in matching
const STOPWORDS = new Set([
  "we","have","an","the","is","are","am","i","you","he","she","it","they",
]);

// ===== LOAD CLASSIFIER ON START =====
natural.BayesClassifier.load("sepitoriClassifier.json", null, (err, loaded) => {
  if (err) console.error("❌ Failed to load model:", err);
  else {
    classifier = loaded;
    console.log("✅ Model loaded and ready!");
  }
});

// =====================
//     PREDICT ROUTE
// =====================
app.post("/predict", (req, res) => {
  if (!classifier) return res.status(503).json({ error: "Model not loaded yet" });

  const { text } = req.body;
  if (!text) return res.status(400).json({ error: "Text is required" });

  const normalized = normalize(text);
  const rawWords = normalized.split(/\s+/).filter(Boolean);
  const stemmedWords = stemWords(rawWords);

  // Remove stopwords
  const filteredWords = stemmedWords.filter((w) => !STOPWORDS.has(w));

  // Model features (stemmed)
  const featureSet = new Set(Object.keys(classifier.features || {}));

  // Detect unseen words
  const unseenWords = filteredWords.filter((w) => !featureSet.has(w));

  // Separate known vs unknown words
  const sepitoriWords = filteredWords.filter((w) => featureSet.has(w));
  const nonSepitoriWords = filteredWords.filter((w) => !featureSet.has(w));

  // Classifier probabilities (optional reference)
  const probs = classifier.getClassifications(normalized);
  const sepProb = probs.find((p) => p.label === "sepitori")?.value || 0;
  const nonProb = probs.find((p) => p.label === "non-sepitori")?.value || 0;
  const total = sepProb + nonProb || 1;
  const sepitoriConfidence = sepProb / total;
  const nonConfidence = nonProb / total;

  // ===== FINAL LABEL LOGIC =====
  let finalLabel = "not recognized";

  if (filteredWords.length === 0) {
    finalLabel = "not recognized";          // nothing meaningful
  } else if (sepitoriWords.length > 0 && nonSepitoriWords.length > 0) {
    finalLabel = "mixed";                   // contains both classes
  } else if (sepitoriWords.length > 0) {
    finalLabel = "sepitori";                // only sepitori
  } else if (nonSepitoriWords.length > 0) {
    finalLabel = "non-sepitori";            // only non-sepitori
  } else {
    finalLabel = "not recognized";          // fallback
  }

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

// =====================
//      TRAIN ROUTE
// =====================
app.post("/train", (req, res) => {
  if (!classifier) return res.status(503).json({ error: "Model not loaded yet" });

  const { text, label } = req.body;
  if (!text || !label)
    return res.status(400).json({ error: "Text and label are required" });

  const normalized = normalize(text);

  // Add to classifier & train
  classifier.addDocument(normalized, label);
  classifier.train();

  classifier.save("sepitoriClassifier.json", (err) => {
    if (err) {
      console.error("❌ Error saving classifier:", err);
      return res.json({ success: false });
    }

    // Reload classifier
    natural.BayesClassifier.load("sepitoriClassifier.json", null, (err, loaded) => {
      if (!err) {
        classifier = loaded;
        console.log("♻️ Classifier reloaded after training!");
      }
    });

    // Update training.js
    const trainingFile = path.join(__dirname, "data", "training.js");
    delete require.cache[require.resolve(trainingFile)];
    let currentData = require(trainingFile);

    currentData.push({ text, label });

    const fileContent =
      "module.exports = " + JSON.stringify(currentData, null, 2) + ";";
    fs.writeFileSync(trainingFile, fileContent, "utf-8");

    res.json({
      success: true,
      message: "New sentence added, model retrained, and reloaded!",
    });
  });
});

// =====================
// DEBUG - See model features
// =====================
app.get("/debug-features", (req, res) => {
  if (!classifier) return res.status(503).json({ error: "Model not loaded" });

  res.json({
    totalFeatures: Object.keys(classifier.features || {}).length,
    featureKeys: Object.keys(classifier.features || {}).slice(0, 200),
  });
});

// ===== SERVER START =====
app.listen(5000, () => {
  console.log("🚀 API runnings on http://localhost:5000");
});
