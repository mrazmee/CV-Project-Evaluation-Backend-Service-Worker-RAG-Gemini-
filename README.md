# 🚀 CV & Project Evaluation Backend Service (Worker + RAG + Gemini)

Backend-only system for automated scoring of a candidate’s **CV** and **Project Report**, aligned with the Study Case rubric.

Uses:
- **Gemini LLM**  
- **Gemini Embeddings**  
- **RAG (Qdrant Cloud)**  
- **Background Worker Queue (SQLite DB)**  
- **PDF Parsing Pipeline**


---

# 📌 1. Core Features

✔ Upload CV & Project Report (PDF)  
✔ Background evaluation job  
✔ RAG rubric retrieval for accurate scoring  
✔ LLM evaluation using structured prompt  
✔ Final score:  
- CV Match Rate (0–1)  
- Project Score (1–5)  
- 3–5 sentence summary  
✔ Store evaluation in SQLite  
✔ Query evaluation result via API

---

# 🏗️ 2. System Architecture

```
Client → POST /upload (save PDF)
Client → POST /evaluate (create job)

Worker:
  - parse PDF
  - embed text (Gemini)
  - retrieve rubric (Qdrant)
  - evaluate (Gemini)
  - save results

Client → GET /result/:id (get final evaluation)
```

---

# ⚙️ 3. Tech Stack

| Layer | Technology |
|------|------------|
| API | Node.js + Express |
| Queue | SQLite (jobs table) |
| DB | SQLite (Prisma ORM) |
| LLM | Gemini 2.0 Flash |
| Embeddings | Gemini embedding model |
| Vector DB | Qdrant Cloud |
| PDF Parsing | pdf-parse |
| Storage | Local uploads folder |

---

# 🔧 4. Environment Variables (.env)

```
# === GEMINI ===
GEMINI_API_KEY=your_key
GEMINI_EMBEDDING_MODEL=gemini-embedding-001

# === QDRANT ===
QDRANT_URL=https://xxxx.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_key
QDRANT_COLLECTION_NAME=rubrics

# === STORAGE ===
UPLOAD_DIR=uploads

# === DATABASE ===
DATABASE_URL=file:./dev.db
```

---

# 📁 5. Project Structure

```
src/
 ├── api/
 │    ├── uploadController.js
 │    └── evaluateController.js
 │    └── resultController.js
 ├── worker/
 │    └── jobWorker.js
 ├── config/
 │    └── env.js
 ├── services/
 │    ├── llmService.js
 │    ├── ragService.js
 │    ├── embeddingService.js
 │    ├── qdrantService.js
 │    ├── fileService.js
 │    ├── pdfService.js
 │    └── jobService.js
 ├── scripts/
 │    └── ingestRubrics.js
data/
 ├── cv_rubrics.txt
 ├── project_rubrics.txt
uploads/
README.md
```

---

# 🚀 6. Running the Project

### Install dependencies
```
npm install
```

### Generate SQLite schema
```
npx prisma migrate dev
```

### Import rubrics into Qdrant
```
npm run ingest:rubrics
```

### Start backend API
```
npm run dev
```

### Start worker (separate terminal)
```
npm run worker
```

---

# 🔌 7. API Endpoints (FINAL & AKTUAL)

## **1) Upload CV & Project Report**
```
POST /upload
Content-Type: multipart/form-data

Fields:
- cv: PDF file
- report: PDF file
```

Response:
```json
{
  "cvFileId": 12,
  "reportFileId": 13
}
```

---

## **2) Create Evaluation Job**
```
POST /evaluate
Content-Type: application/json
```

Body:
```json
{
  "cvFileId": 12,
  "reportFileId": 13,
  "jobTitle": "Backend Intern"
}
```

Response:
```json
{
  "jobId": 24,
  "status": "queued"
}
```

---

## **3) Get Evaluation Result**
```
GET /result/:jobId
```

Example:
```json
{
  "jobId": 24,
  "cvMatchRate": 0.71,
  "projectScore": 4.9,
  "overallSummary": "Candidate demonstrates strong backend potential...",
  "usedFallback": false
}
```

---

# 🧠 8. How RAG Works

1. Parse PDF → extract text  
2. Gemini embedding → 768 dim  
3. Qdrant search `"cv_rubric"` or `"project_rubric"`  
4. Inject retrieved rubrics into LLM prompt  
5. LLM generates evaluation  
6. System applies weighting and returns final score  

Ensures **consistent, rubric-aligned scoring**.

---

# 📊 9. Evaluation Formula

## CV Rating (1–5)
Weighted components:
- Skills match — 40%
- Experience relevance — 25%
- Achievements/impact — 20%
- Soft skills — 15%

---

## Project Score (1–5)
Weighted components:
- Correctness & architecture — 30%
- Code quality — 25%
- Resilience — 20%
- Documentation quality — 15%
- Bonus creativity — 10%

---

## Overall Summary
Always **3–5 sentences**.  
Summarizes:
- Candidate strengths  
- Weaknesses  
- Fit for the internship role  

---

# ⚠️ 10. Limitations

- Worker is single-threaded  
- No retry for Gemini/Qdrant  
- No authentication  
- Free Gemini quota may run out  
- PDF parsing may fail for scanned documents  

---

# 🚀 11. Future Improvements

- Redis + BullMQ queue  
- Retry & backoff strategy  
- Add Swagger API Docs  
- Deploy via Docker → Render / Railway / Cloud Run  
- Add frontend dashboard  

---

# ✔️ 12. Status

Backend is **complete**, fully functional, and aligned with the study case rubric.

