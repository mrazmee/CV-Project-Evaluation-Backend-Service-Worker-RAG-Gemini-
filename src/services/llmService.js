// src/services/llmService.js
require("dotenv").config();
const { GoogleGenAI } = require("@google/genai");
const { findRubricsForCvAndProject } = require("./ragService");

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const MODEL_NAME = process.env.GEMINI_MODEL || "gemini-2.5-flash";

const genAI = new GoogleGenAI({
  apiKey: GEMINI_API_KEY,
});

// ---------- helper: bersihin output JSON dari ```json ... ``` ----------
function cleanJsonText(raw) {
  if (!raw) return "";
  let text = String(raw).trim();

  // Kalau dibungkus ```json ... ```
  if (text.startsWith("```")) {
    // buang baris pertama ``` / ```json
    text = text.replace(/^```[a-zA-Z]*\s*/, "");
    // buang ``` terakhir
    text = text.replace(/```$/, "").trim();
  }

  // Ambil cuma blok {...} pertama–terakhir
  const firstBrace = text.indexOf("{");
  const lastBrace = text.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    text = text.slice(firstBrace, lastBrace + 1);
  }

  return text;
}

// ---------- helper: bikin prompt ----------
function buildPrompt({
  jobTitle,
  cvText,
  reportText,
  cvRubricsText,
  projectRubricsText,
}) {
  return `
You are evaluating a candidate for a backend / IT internship. Use the provided rubrics and return a STRICT JSON object.

Vacancy title: ${jobTitle || "-"}

=== RAG CV RUBRICS (guidelines) ===
${cvRubricsText || "-"}

=== RAG PROJECT RUBRICS (guidelines) ===
${projectRubricsText || "-"}

=== CV TEXT (raw) ===
${cvText || "(empty)"}

=== PROJECT REPORT TEXT (raw) ===
${reportText || "(empty)"}

SCORING RULES (jangan tulis skor total, cukup sub-score di JSON):

1) CV scoring – berikan sub-score 1–5 (number) untuk:
   - technical   : kemampuan teknis backend/IT, tools, stack.
   - experience  : relevansi pengalaman dengan role.
   - achievements: pencapaian terukur, impact, hasil nyata.
   - culture     : kolaborasi, komunikasi, learning attitude, ownership.

   Nanti sistem akan menghitung sendiri weighted score:
   - technical   = 40%
   - experience  = 25%
   - achievements= 20%
   - culture     = 15%

2) PROJECT scoring – berikan sub-score 1–5 (number) untuk:
   - correctness   : ketepatan solusi, implementasi, kebenaran teknis.
   - structure     : arsitektur, modularitas, clean code / desain.
   - resilience    : error handling, security, scalability, reliability.
   - documentation : kejelasan penjelasan, struktur laporan, kemudahan dibaca.
   - creativity    : orisinalitas solusi, pemilihan teknologi, insight tambahan.

   Nanti sistem akan menghitung sendiri overall projectScore (1–5) dari:
   - correctness   = 30%
   - structure     = 25%
   - resilience    = 20%
   - documentation = 15%
   - creativity    = 10%

3) OUTPUT FORMAT (PENTING):
Kembalikan HANYA JSON mentah (tanpa markdown, tanpa backticks), dengan schema PERSIS seperti ini:

{
  "cvScores": {
    "technical": 1-5 (number),
    "experience": 1-5 (number),
    "achievements": 1-5 (number),
    "culture": 1-5 (number)
  },
  "projectScores": {
    "correctness": 1-5 (number),
    "structure": 1-5 (number),
    "resilience": 1-5 (number),
    "documentation": 1-5 (number),
    "creativity": 1-5 (number)
  },
  "cvFeedback": "paragraf singkat dalam Bahasa Indonesia",
  "projectFeedback": "paragraf singkat dalam Bahasa Indonesia",
  "overallSummary": "ringkasan singkat dalam Bahasa Indonesia"
}

JANGAN tambahkan teks lain di luar JSON.
`;
}

// ---------- helper: retry kalau 429 / 503 ----------
async function callGeminiWithRetry(prompt, maxRetries = 3) {
  let lastError;
  const contents = [
    {
      role: "user",
      parts: [{ text: prompt }],
    },
  ];

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await genAI.models.generateContent({
        model: MODEL_NAME,
        contents,
      });
      return response;
    } catch (err) {
      lastError = err;
      if ((err.status === 503 || err.status === 429) && attempt < maxRetries) {
        const delayMs = 1000 * attempt; // 1s, 2s, 3s
        await new Promise((res) => setTimeout(res, delayMs));
        continue;
      }
      throw err;
    }
  }

  throw lastError;
}

// ---------- fallback heuristic ----------
function heuristicFallback({ jobTitle, cvText, reportText }) {
  const cvLen = cvText ? cvText.length : 0;
  const reportLen = reportText ? reportText.length : 0;

  const cvMatchRate = Math.max(0.2, Math.min(0.9, cvLen / 8000));
  const projectScore = Math.max(1, Math.min(5, reportLen / 1500));

  return {
    cvMatchRate: parseFloat(cvMatchRate.toFixed(2)),
    projectScore: parseFloat(projectScore.toFixed(2)),
    cvFeedback:
      "Automatic fallback evaluation: CV terlihat cukup relevan, namun penilaian ini tidak berasal dari model LLM karena terjadi kegagalan saat memanggil API.",
    projectFeedback:
      "Automatic fallback evaluation: Project report dinilai berdasarkan panjang dan struktur dasar teks. Disarankan untuk melakukan review manual.",
    overallSummary:
      "Sistem gagal memanggil LLM dan menggunakan heuristic fallback. Untuk keputusan rekrutmen sebaiknya dilakukan penilaian manual tambahan.",
    usedFallback: true,
  };
}

// ---------- fungsi utama dipakai worker ----------
async function evaluateCandidate({ jobTitle, cvText, reportText }) {
  const safeCv = cvText || "";
  const safeReport = reportText || "";

  const fallback = () =>
    heuristicFallback({ jobTitle, cvText: safeCv, reportText: safeReport });

  if (!GEMINI_API_KEY) {
    console.warn("[llmService] GEMINI_API_KEY tidak diset, pakai fallback.");
    return fallback();
  }

  // --- ambil rubrics dari RAG (Qdrant) ---
  let cvRubricsText = "";
  let projectRubricsText = "";

  try {
    const rag = await findRubricsForCvAndProject({
      cvText: safeCv,
      reportText: safeReport,
    });

    cvRubricsText = rag.cvRubricsText || "";
    projectRubricsText = rag.projectRubricsText || "";

    console.log(
      `[llmService] RAG rubrics: cvLen=${cvRubricsText.length}, projectLen=${projectRubricsText.length}`
    );
  } catch (err) {
    console.warn("[llmService] Gagal mengambil rubrics dari RAG:", err);
  }

  const prompt = buildPrompt({
    jobTitle,
    cvText: safeCv,
    reportText: safeReport,
    cvRubricsText,
    projectRubricsText,
  });

  try {
    const result = await callGeminiWithRetry(prompt);

    // 🔧 Bagian yang diperbaiki: cara ambil teks dari response
    let rawText = "";

    if (typeof result.response?.text === "function") {
      // Pola: result.response.text()
      rawText = await result.response.text();
    } else if (typeof result.text === "function") {
      // Pola: result.text()
      rawText = await result.text();
    } else if (Array.isArray(result.candidates)) {
      // Pola: candidates[].content.parts[].text
      rawText = result.candidates
        .flatMap((c) => c.content?.parts || [])
        .map((p) => p.text || "")
        .join("\n");
    } else {
      console.warn(
        "[llmService] Tidak menemukan text di response, pakai fallback. Raw:",
        JSON.stringify(result, null, 2)
      );
      return fallback();
    }

    const cleaned = cleanJsonText(rawText);

    let parsed;
    try {
      parsed = JSON.parse(cleaned);
    } catch (e) {
      console.warn(
        "[llmService] Response bukan JSON valid, pakai fallback. Raw:",
        rawText
      );
      return fallback();
    }

    const cvScores = parsed.cvScores || {};
    const projectScores = parsed.projectScores || {};

    const cvTechnical = Number(cvScores.technical) || 0;
    const cvExperience = Number(cvScores.experience) || 0;
    const cvAchievements = Number(cvScores.achievements) || 0;
    const cvCulture = Number(cvScores.culture) || 0;

    const projectCorrectness = Number(projectScores.correctness) || 0;
    const projectStructure = Number(projectScores.structure) || 0;
    const projectResilience = Number(projectScores.resilience) || 0;
    const projectDocumentation = Number(projectScores.documentation) || 0;
    const projectCreativity = Number(projectScores.creativity) || 0;

    // --- hitung weighted score pakai rubric (CV: 1–5 -> 0–1, Project: 1–5) ---
    const cvWeighted =
      cvTechnical * 0.4 +
      cvExperience * 0.25 +
      cvAchievements * 0.2 +
      cvCulture * 0.15;

    const projectWeighted =
      projectCorrectness * 0.3 +
      projectStructure * 0.25 +
      projectResilience * 0.2 +
      projectDocumentation * 0.15 +
      projectCreativity * 0.1;

    const cvMatchRate = parseFloat((cvWeighted / 5).toFixed(2)); // 0–1
    const projectScore = parseFloat(projectWeighted.toFixed(2)); // 1–5

    return {
      cvMatchRate,
      cvFeedback: parsed.cvFeedback || "",
      projectScore,
      projectFeedback: parsed.projectFeedback || "",
      overallSummary: parsed.overallSummary || "",
      usedFallback: false,
    };
  } catch (err) {
    console.error("[llmService] LLM call failed, using fallback:", err);
    return fallback();
  }
}

module.exports = {
  evaluateCandidate,
};
