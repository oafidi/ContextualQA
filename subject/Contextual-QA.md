# Project Specification – Contextual Question Answering (Contextual QA) in Moroccan Darija
**Design of an NLP Pipeline for Text Comprehension and Automated Question Answering from Context**

---

## 1. General Context

**Contextual Question Answering (QA)** is an advanced Natural Language Processing (NLP) task. It consists of automatically answering a **question in natural language** by explicitly relying on a **provided context text** in which the sought information is found.

In the case of **Moroccan Darija**, this task is particularly complex due to:
- the absence of annotated QA corpora;
- orthographic and syntactic variability;
- code-switching (Darija / French / Modern Standard Arabic);
- the scarcity of adapted pre-trained models.

This project aims to expose students to a **real QA use case for a low-resource language**, with emphasis on:
- **building an annotated contextual QA corpus**;
- **LLM-assisted semi-automatic annotation**;
- comparison of different Question Answering approaches.

---

## 2. Project Objectives

### 2.1 Pedagogical Objectives

Upon completion of the project, the student should be able to:
- collect context texts in Darija;
- formulate and structure **question–answer** pairs;
- design an NLP pipeline dedicated to contextual QA;
- implement **semi-automatic annotation of QA data**;
- train and compare several Question Answering approaches;
- critically analyze system performance and errors.

### 2.2 Target Skills

- NLP applied to low-resource languages
- Contextual Question Answering
- Structured data annotation (text, question, answer)
- Machine Learning / LLM
- Reproducibility and experimental tracking

---

## 3. Scope and Constraints

- Project carried out in **pairs**
- Main language: **Python**
- Target language: **Moroccan Darija**
- Data sourced **exclusively from public sources**
- Mandatory use of:
  - **semi-automatic annotation (LLM + human validation)**;
  - an **experimental tracking tool** (MLflow or Weights & Biases).
- Technical report and documented code are required.

---

## 4. Definition of the Contextual Question Answering Task

### 4.1 Nature of the Task

The task consists of developing a system capable of answering a **question in Darija** by relying on a **Darija context text**, explicitly provided as input.

Each dataset instance is structured as follows:
(context, question) → answer

The answer must:
- be **contained in or inferable from the context**;
- be concise and relevant;
- not introduce information external to the text.

---

### 4.2 Type of QA Adopted

A QA instance is defined as:

> A **clear and unambiguous question**, formulated in Darija,  
> whose **answer is an exact substring of the context**.

#### Absolute Constraints

- ❌ no freely generated answers
- ❌ no paraphrased answers
- ❌ no answers from external knowledge
- ✅ the answer must be **copied word for word from the context**

### 4.3 Allowed Question Types

Questions must address **facts explicitly mentioned** in the first paragraph:

### Recommended Types
- Who? (person, institution)
- Where? (specific location)
- When? (date, period)
- What? (event, decision, action)
- How many? (figures, quantities)
- Which? (unique entity)

### Prohibited Types
- Why?
- How?
- Implicit or interpretative questions
- Value judgments
- Questions requiring external inference

---

### 4.4 Processing Unit

- Context: paragraphs or short articles (100 to 500 words)
- Question: interrogative sentence in Darija
- Answer: word, word group, or short phrase

---

### 4.5 Darija-Specific Linguistic Constraints

Developed systems must explicitly account for:
- orthographic variation;
- Darija / FR / AR code-switching;
- use of Arabic or Latin alphabet (Arabizi);
- oral and elliptical question formulations.

These aspects must be analyzed in the error evaluation.

---

## 5. Data Sources and QA Corpus Construction

The source data has already been scraped and is provided to students as supplementary files.
No new data collection is required.

### 5.1 Goud.ma Articles

- **Source**: https://www.goud.ma
- **Period covered**: September 2024 → January 2026
- **Language**: Moroccan Darija (mainly dialectal Arabic)
- **Content types**:
  - national news,
  - miscellaneous facts,
  - politics,
  - society,
  - economics,
  - sports,
  - culture.

Articles are considered **information documents**, suited to contextual QA thanks to their narrative and factual structure.

---

## 6. Step 1 – Preprocessing and Exploration

### 6.1 Fundamental Principle

For each Goud.ma article:

> **Only the first paragraph of the article is used as context.**

### Justification
- The first paragraph generally contains:
  - a summary of essential facts,
  - key entities (persons, places, dates),
  - the highest semantic density information.
- This reduces:
  - context length,
  - ambiguities,
  - the number of possible answers.

### 6.2 Definition of the `context` Field

The `context` field corresponds exactly to:
- the **raw first paragraph**,
- without reformulation,
- without heavy normalization,
- without removal of information.

### 6.3 Preprocessing

- light cleaning (URLs, emojis, spam);
- minimal and justified normalization;
- clear context segmentation.

### 6.4 Exploratory Analysis

- context lengths;
- question types (who, what, where, why, how);
- answer distribution;
- code-switching analysis.

---

## 7. Step 2 – Semi-Automatic QA Annotation (Central Step)

### 7.1 Objective

Build a reliable QA corpus by combining:
- automatic generation of questions and answers by LLM;
- targeted human validation and correction;
- systematic quality control.

---

### 7.2 Recommended Annotation Approach

#### Step 1 – Manually Annotated Reference Set

- Select **800 to 1,500 contexts**.
- Manually write questions and answers.
- Define a **QA annotation guide**.
- Constitute a **partial gold standard**.

---

#### Step 2 – Automatic Generation by LLM

##### Recommended Model

* **Gemini 2.5** or **Gemini 3**
* Access via:

  * Google API
  * Google Colab

---

##### Prompt (Extractive QA)

Students must use a **strictly constrained** prompt, for example:

```
Given the following Darija news paragraph, generate clear and unambiguous
questions in Darija such that the answer is explicitly present in the text.

IMPORTANT RULES:
- The answer MUST be copied verbatim from the context.
- The answer must be unique in the context.
- Do NOT paraphrase the answer.
- Do NOT invent information.

Output JSON objects using this format ONLY:
{"context": "...", "question": "...", "answer": "..."}

Here is the context:
{text}
```

---

#### Step 3 – Assisted Validation with Label Studio

The triplets (context, question, answer) must be imported into **Label Studio**.

Human annotators must:
- validate the relevance of the question;
- verify the accuracy of the answer;
- correct or reject incorrect triplets.

---

#### Step 4 – Quality Control

- correction rate of generated pairs;
- analysis of frequent errors (ambiguous questions, out-of-context answers);
- harmonization of the final corpus.

⚠️ **Human validation is mandatory and non-optional.**

Annotators must:

- Delete
  * ambiguous questions,
  * answers appearing multiple times in the context,
  * questions whose answer requires interpretation.
- Correct
  * minor linguistic errors in the question,
  * incorrect selection of the answer span.
- Systematically verify
  * uniqueness of the answer,
  * character-by-character exact match,
  * question ↔ answer consistency.

##### Correct Example

**Context**
```
قالت مصادر مطلعة أن المحكمة الابتدائية بالدار البيضاء قررت تأجيل
محاكمة المتهم إلى غاية 15 أكتوبر 2024.
```

**QA**
```json
{
  "context": "قالت مصادر مطلعة أن المحكمة الابتدائية بالدار البيضاء قررت تأجيل محاكمة المتهم إلى غاية 15 أكتوبر 2024.",
  "question": "إلى غاية شحال تأجلات المحاكمة؟",
  "answer": "15 أكتوبر 2024"
}
```

---

### Incorrect Example (answer not verbatim)

```json
{
  "question": "فين كانت المحكمة؟",
  "answer": "في الدار البيضاء"
}
```

❌ The answer **is not an exact substring** of the context.

---

##### Incorrect Example (ambiguity)

```json
{
  "question": "شكون المعني بالقضية؟",
  "answer": "المتهم"
}
```

❌ Multiple entities could correspond.

#### Corpus Quality Criteria

The final corpus must satisfy:

* 100% of answers are extractive;
* no answer is freely generated;
* each question admits **one single correct answer**;
* the context is sufficient to answer without external knowledge.

---

## 8. Step 3 – Contextual QA Modeling

The goal is to compare the ability of different architectures to precisely identify the "span" (text segment) containing the answer within a Darija context.

### 8.1 Expected Approaches

#### A. Baseline: Classical Machine Learning

Before using heavy models, students must establish a comparison point with traditional information retrieval methods:

* **Term Similarity (TF-IDF / BM25):** Computing the relevance between the question and context sentences to extract the most probable sentence.
* **Static Embeddings:** Using word vectors (e.g., FastText pre-trained on Maghrebi data) to compute average cosine similarity between the question and the context.

#### B. Supervised Extractive QA (Transformers)

This is the preferred approach for modern NLP. It consists of predicting the start and end indices of the answer.

* **DarijaBERT:** Use of specialized language models like **DarijaBERT** or **MarBERT**. These models, pre-trained on billions of Arabic dialect tokens, are particularly robust against Darija's specificities.
* **Multi-script Handling:** Evaluation of the model's ability to process both Arabic script and **Arabizi** (Latin characters), very present in collected Reddit and YouTube data.

---

### 8.2 Modeling-Specific Preprocessing

* **Tokenization:** Adapting the tokenizer to avoid breaking Darija's complex morphological structures.
* **Normalization:** Strategic choice (or not) to normalize certain characters (e.g., 3, 7, 9 in Arabizi) before injection into BERT models.

---

### 8.3 Evaluation

- Exact Match (EM)
- F1-score
- Qualitative analysis of incorrect answers:
  * **Boundary Errors:** The model finds the answer but includes superfluous words before or after.
  * **Type Errors:** The model gives a name when a number is expected (e.g., "شحال" → "بزاف" instead of "10").
  * **Code-switching Impact:** Performance analysis depending on whether the question/answer is in Arabic or Arabizi characters.
  * **Hallucinations (if generative approach tested):** Cases where the answer is not present in the provided context.

---

## 9. Step 4 – Experimental Tracking

All experiments must be tracked using:
- **MLflow**, or
- **Weights & Biases**.

Each run must include:
- corpus version;
- QA model type;
- parameters;
- obtained metrics.

---

## 10. Expected Deliverables

1. Annotated contextual QA corpus in Darija
2. Detailed technical report
3. Structured and documented source code
4. Experimental tracking dashboard
5. Qualitative QA error analysis

---

## 11. Evaluation Criteria (Indicative)

| Criterion | Weight |
|------|-------------|
| Corpus quality and annotation | 50% |
| NLP / QA Pipeline | 15% |
| Modeling and evaluation | 10% |
| Tracking and reproducibility | 10% |
| Critical analysis and report | 15% |

---

## 12. Possible Extensions (Bonus)

- Multi-document QA
- Extractive vs. generative comparison
- Study of the impact of context length
- QA + Darija chatbot integration

---