// src/services/llmService.js
require("dotenv").config();
const { GoogleGenAI } = require("@google/genai");
const { findRubricsForCvAndProject } = require("./ragService");

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const MODEL_NAME = process.env.GEMINI_MODEL || "gemini-2.5-flash";

// --- client Gemini ---
const genAI = new GoogleGenAI({
  apiKey: GEMINI_API_KEY,
});

// ----------------- helper: clamp & rounding -----------------
function clampScore(x, min = 1, max = 5) {
  const n = Number(x);
  if (Number.isNaN(n)) return min;
  return Math.min(max, Math.max(min, n));
}

function round2(x) {
  return Math.round((Number(x) || 0) * 100) / 100;
}

// ----------------- build prompt (sesuai rubric PDF) --------
function buildPrompt({ jobTitle, cvText, reportText, cvRubricsText, projectRubricsText }) {
  return `
You are an AI evaluator for internship candidates.

Vacancy title: ${jobTitle || "-"}

========================================
RAW CV TEXT
========================================
${cvText || "(empty)"}

========================================
RAW PROJECT REPORT TEXT
========================================
${reportText || "(empty)"}

========================================
SCORING RUBRICS (from company PDF)
========================================

1) CV Match Evaluation (1–5 per parameter, weighted)

Parameters & Weights:
- technicalSkills (Weight: 40%)
  Alignment with job requirements (backend, databases, APIs, cloud, AI/LLM).
  Scoring:
    1 = Irrelevant skills
    2 = Few overlaps
    3 = Partial match
    4 = Strong match
    5 = Excellent match + AI/LLM exposure

- experienceLevel (Weight: 25%)
  Years of experience & project complexity.
  Scoring:
    1 = <1 yr / trivial projects
    2 = 1–2 yrs
    3 = 2–3 yrs w/ mid-scale projects
    4 = 3–4 yrs solid track record
    5 = 5+ yrs / high-impact projects

- relevantAchievements (Weight: 20%)
  Impact of past work (scaling, performance, adoption).
  Scoring:
    1 = No clear achievements
    2 = Minimal improvements
    3 = Some measurable outcomes
    4 = Significant contributions
    5 = Major measurable impact

- culturalFit (Weight: 15%)
  Communication, learning mindset, teamwork/leadership.
  Scoring:
    1 = Not demonstrated
    2 = Minimal
    3 = Average
    4 = Good
    5 = Excellent & well demonstrated

2) Project Deliverable Evaluation (1–5 per parameter, weighted)

Parameters & Weights:
- correctness (Weight: 30%)
  Prompt design, LLM chaining or ML integration, RAG / context injection, end-to-end logic correctness.

- codeQuality (Weight: 25%)
  Clean, modular, reusable, tested. Code organization consistent with best practices.

- resilience (Weight: 20%)
  Handling long jobs, retries, randomness, API failures, logging, stability.

- documentation (Weight: 15%)
  README clarity, setup instructions, trade-off explanations.

- creativity (Weight: 10%)
  Extra features beyond requirements, thoughtful enhancements.

3) Overall Candidate Evaluation

- CV Score (1–5): weighted average of CV parameters.
- CV Match Rate (0–1): CV Score converted to decimal via CV_Score * 0.2.
- Project Score (1–5): weighted average of project parameters.
- Overall Summary: 3–5 sentences describing strengths, gaps, and recommendations.

========================================
EXTRA RUBRICS CONTEXT FROM KNOWLEDGE BASE (RAG)
========================================

[CV Rubrics Hints]
${cvRubricsText || "(none)"}

[Project Rubrics Hints]
${projectRubricsText || "(none)"}

========================================
TASK
========================================

1. Carefully read the CV and project report.
2. Score each parameter from 1 to 5 (integers only).
3. Use the exact weights listed above to compute:
   - cvScore: weighted average of CV parameters (1–5)
   - projectScore: weighted average of project parameters (1–5)
   - cvMatchRate: cvScore * 0.2  (range 0.0–1.0)
4. Write short reasons for each parameter score.
5. Write an overallSummary of **3–5 sentences** in a professional tone
   (include strengths, weaknesses/gaps, and recommendations).

IMPORTANT:
- All parameter scores MUST be integers 1, 2, 3, 4, or 5.
- cvScore and projectScore MUST be between 1 and 5.
- cvMatchRate MUST be between 0.0 and 1.0.
- Do NOT mention the internal rubric or weights in the summary.

Return ONLY valid JSON with this EXACT schema (no additional text):

{
  "cv": {
    "technicalSkills": { "score": 1-5, "reason": "string" },
    "experienceLevel": { "score": 1-5, "reason": "string" },
    "relevantAchievements": { "score": 1-5, "reason": "string" },
    "culturalFit": { "score": 1-5, "reason": "string" }
  },
  "project": {
    "correctness": { "score": 1-5, "reason": "string" },
    "codeQuality": { "score": 1-5, "reason": "string" },
    "resilience": { "score": 1-5, "reason": "string" },
    "documentation": { "score": 1-5, "reason": "string" },
    "creativity": { "score": 1-5, "reason": "string" }
  },
  "overallSummary": "3–5 sentences string",
  "cvScore": 1-5,
  "cvMatchRate": 0.0-1.0,
  "projectScore": 1-5
}
`;
}

// ----------------- heuristic fallback -----------------
function heuristicFallback({ jobTitle, cvText, reportText }) {
  const cvLen = cvText ? cvText.length : 0;
  const reportLen = reportText ? reportText.length : 0;

  // Kasar: makin panjang teks, makin tinggi sedikit skornya
  const roughCv = clampScore(2 + Math.min(3, cvLen / 3000));       // 2–5
  const roughProj = clampScore(2 + Math.min(3, reportLen / 3000));  // 2–5

  return {
    cvMatchRate: round2(roughCv * 0.2),        // 0–1
    cvScore: round2(roughCv),                  // 1–5
    projectScore: round2(roughProj),           // 1–5
    cvFeedback:
      "Automatic fallback evaluation: CV dianggap cukup relevan berdasarkan panjang dan struktur umum, tetapi penilaian ini tidak berasal dari model LLM.",
    projectFeedback:
      "Automatic fallback evaluation: Project report dianggap cukup baik berdasarkan panjang dan struktur teks, namun disarankan melakukan review manual.",
    overallSummary:
      "Sistem gagal memanggil LLM dan menggunakan heuristic fallback sederhana berdasarkan panjang dokumen. Untuk keputusan rekrutmen, sebaiknya dilakukan penilaian manual tambahan terhadap CV dan project report.",
    rawCvScores: null,
    rawProjectScores: null,
    usedFallback: true,
  };
}

// ----------------- panggilan Gemini dengan retry --------
async function callGeminiWithRetry(prompt, maxRetries = 3) {
  let lastError;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await genAI.models.generateContent({
        model: MODEL_NAME,
        contents: [{ role: "user", parts: [{ text: prompt }] }],
      });
      return response;
    } catch (err) {
      lastError = err;
      const status = err?.status || err?.code;

      // 503 / 429 → retry dengan backoff sederhana
      if ((status === 503 || status === 429) && attempt < maxRetries) {
        const delayMs = 1000 * attempt;
        await new Promise((res) => setTimeout(res, delayMs));
        continue;
      }

      throw err;
    }
  }

  throw lastError;
}

// ----------------- parsing JSON dari Gemini -------------
function extractTextFromGeminiResponse(response) {
  if (!response) return "";

  // Beberapa versi SDK punya .text
  if (typeof response.text === "string" && response.text.trim()) {
    return response.text;
  }

  // Versi lain: candidates[0].content.parts[0].text
  const partText =
    response.candidates?.[0]?.content?.parts?.[0]?.text ||
    response.candidates?.[0]?.content?.parts?.map((p) => p.text).join("\n");

  return partText || "";
}

function parseGeminiJson(rawText) {
  if (!rawText) {
    throw new Error("Empty response from LLM");
  }

  // Hilangkan ```json ... ```
  const cleaned = rawText
    .replace(/```json/gi, "")
    .replace(/```/g, "")
    .trim();

  return JSON.parse(cleaned);
}

// ----------------- perhitungan score dari JSON ----------
function computeScoresFromParsed(parsed) {
  // Ambil sub-objek (jaga-jaga kalau LLM skip)
  const cv = parsed.cv || {};
  const proj = parsed.project || {};

  // CV parameter (1–5)
  const cvScores = {
    technicalSkills: clampScore(cv.technicalSkills?.score),
    experienceLevel: clampScore(cv.experienceLevel?.score),
    relevantAchievements: clampScore(cv.relevantAchievements?.score),
    culturalFit: clampScore(cv.culturalFit?.score),
  };

  // Project parameter (1–5)
  const projectScores = {
    correctness: clampScore(proj.correctness?.score),
    codeQuality: clampScore(proj.codeQuality?.score),
    resilience: clampScore(proj.resilience?.score),
    documentation: clampScore(proj.documentation?.score),
    creativity: clampScore(proj.creativity?.score),
  };

  // Weighting dari rubric PDF
  const cvWeights = {
    technicalSkills: 0.4,
    experienceLevel: 0.25,
    relevantAchievements: 0.2,
    culturalFit: 0.15,
  };

  const projectWeights = {
    correctness: 0.3,
    codeQuality: 0.25,
    resilience: 0.2,
    documentation: 0.15,
    creativity: 0.1,
  };

  // Weighted averages
  const cvScore =
    cvScores.technicalSkills * cvWeights.technicalSkills +
    cvScores.experienceLevel * cvWeights.experienceLevel +
    cvScores.relevantAchievements * cvWeights.relevantAchievements +
    cvScores.culturalFit * cvWeights.culturalFit;

  const projectScore =
    projectScores.correctness * projectWeights.correctness +
    projectScores.codeQuality * projectWeights.codeQuality +
    projectScores.resilience * projectWeights.resilience +
    projectScores.documentation * projectWeights.documentation +
    projectScores.creativity * projectWeights.creativity;

  // Clamp just in case
  const cvScoreFinal = clampScore(cvScore);
  const projectScoreFinal = clampScore(projectScore);

  // CV match rate 0–1 (×0.2)
  const cvMatchRate = round2(cvScoreFinal * 0.2);

  // Reason text / feedback
  const cvFeedback =
    parsed.cvFeedback ||
    parsed.cv_feedback ||
    parsed.cvComment ||
    parsed.cvSummary ||
    "CV evaluation available in parameter reasons, but no explicit cvFeedback field was provided.";

  const projectFeedback =
    parsed.projectFeedback ||
    parsed.project_feedback ||
    parsed.projectComment ||
    "Project evaluation available in parameter reasons, but no explicit projectFeedback field was provided.";

  const overallSummary =
    parsed.overallSummary ||
    parsed.summary ||
    "Overall summary not provided explicitly by the model.";

  const result = {
    cvMatchRate,
    cvScore: round2(cvScoreFinal),
    projectScore: round2(projectScoreFinal),
    cvFeedback,
    projectFeedback,
    overallSummary,
    rawCvScores: cvScores,
    rawProjectScores: projectScores,
    usedFallback: false,
  };

  console.log("[llmService] Normalized scores:", {
    cvScore: result.cvScore,
    cvMatchRate: result.cvMatchRate,
    projectScore: result.projectScore,
  });

  return result;
}

// ----------------- fungsi utama (dipakai worker) --------
async function evaluateCandidate({ jobTitle, cvText, reportText }) {
  const cv = cvText || "";
  const project = reportText || "";

  // Kalau dua-duanya kosong, langsung return tanpa ke LLM
  if (!cv.trim() && !project.trim()) {
    return {
      cvMatchRate: 0,
      cvScore: 0,
      projectScore: 0,
      cvFeedback:
        "No CV text was provided, so an evaluation cannot be performed.",
      projectFeedback:
        "No project report text was provided, so an evaluation cannot be performed.",
      overallSummary:
        "Neither CV nor project report content was provided. Please submit both documents for assessment.",
      rawCvScores: null,
      rawProjectScores: null,
      usedFallback: false,
    };
  }

  if (!GEMINI_API_KEY) {
    console.warn("[llmService] GEMINI_API_KEY tidak diset, pakai fallback.");
    return heuristicFallback({ jobTitle, cvText: cv, reportText: project });
  }

  // 1) Ambil rubrics dari RAG
  let cvRubricsText = "";
  let projectRubricsText = "";

  try {
    const rag = await findRubricsForCvAndProject({
      cvText: cv,
      reportText: project,
    });
    cvRubricsText = rag.cvRubricsText || "";
    projectRubricsText = rag.projectRubricsText || "";
    console.log(
      "[llmService] RAG rubrics: cvLen=%d, projectLen=%d",
      cvRubricsText.length,
      projectRubricsText.length
    );
  } catch (err) {
    console.warn("[llmService] Gagal mengambil rubrics dari RAG:", err);
  }

  const prompt = buildPrompt({
    jobTitle,
    cvText: cv,
    reportText: project,
    cvRubricsText,
    projectRubricsText,
  });

  try {
    const response = await callGeminiWithRetry(prompt);
    const rawText = extractTextFromGeminiResponse(response);

    let parsed;
    try {
      parsed = parseGeminiJson(rawText);
    } catch (e) {
      console.warn(
        "[llmService] Response bukan JSON valid, pakai fallback. Raw:",
        rawText
      );
      return heuristicFallback({ jobTitle, cvText: cv, reportText: project });
    }

    return computeScoresFromParsed(parsed);
  } catch (err) {
    console.error("[llmService] LLM call failed, using fallback:", err);
    return heuristicFallback({ jobTitle, cvText: cv, reportText: project });
  }
}

module.exports = {
  evaluateCandidate,
};
