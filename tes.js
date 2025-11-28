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

// Normalize + stem words
const normalize = (text) =>
  text.toLowerCase().replace(/[^\w\s]/g, "");

const stemWords = (words) => words.map((w) => stemmer.stem(w));

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
  if (!classifier)
    return res.status(503).json({ error: "Model not loaded yet" });

  const { text } = req.body;
  if (!text) return res.status(400).json({ error: "Text is required" });

  const normalized = normalize(text);
  const rawWords = normalized.split(/\s+/).filter(Boolean);
  const stemmedWords = stemWords(rawWords);

  // Model features (stemmed)
  const modelWords = Object.keys(classifier.features || {});

  // Detect unseen (stem-based)
  const unseenWords = stemmedWords.filter((w) => !modelWords.includes(w));

  // Get probabilities
  const probs = classifier.getClassifications(normalized);
  const sepProb = probs.find((p) => p.label === "sepitori")?.value || 0;
  const nonProb = probs.find((p) => p.label === "non-sepitori")?.value || 0;

  const total = sepProb + nonProb || 1;
  const sepitoriConfidence = sepProb / total;
  const nonConfidence = nonProb / total;

  // ===== FINAL LABEL LOGIC =====
  let finalLabel = "not recognized";

  if (stemmedWords.length > 0 && unseenWords.length === stemmedWords.length) {
    finalLabel = "not recognized";
  } else if (nonConfidence >= 0.65) {
    finalLabel = "non-sepitori";
  } else if (sepitoriConfidence >= 0.65) {
    finalLabel = "sepitori";
  } else {
    finalLabel = "mixed";
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
  if (!classifier)
    return res.status(503).json({ error: "Model not loaded yet" });

  const { text, label } = req.body;
  if (!text || !label)
    return res.status(400).json({ error: "Text and label are required" });

  const normalized = normalize(text);

  classifier.addDocument(normalized, label);
  classifier.train();

  classifier.save("sepitoriClassifier.json", (err) => {
    if (err) {
      console.error("❌ Error saving classifier:", err);
      return res.json({ success: false });
    }

    natural.BayesClassifier.load("sepitoriClassifier.json", null, (err, loaded) => {
      if (!err) {
        classifier = loaded;
        console.log("♻️ Classifier reloaded after training!");
      }
    });

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

// DEBUG - See what words classifier knows
app.get("/debug-features", (req, res) => {
  if (!classifier)
    return res.status(503).json({ error: "Model not loaded" });

  res.json({
    totalFeatures: Object.keys(classifier.features || {}).length,
    featureKeys: Object.keys(classifier.features || {}).slice(0, 200),
  });
});

// ===== SERVER START =====
app.listen(5000, () => {
  console.log("🚀 API running on http://localhost:5000");
});
