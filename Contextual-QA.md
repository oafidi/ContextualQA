# Cahier des charges du projet Question–Réponse contextuelle (Contextual QA) en Darija marocaine  
**Conception d’une pipeline NLP pour la compréhension de texte et la réponse automatique à partir d’un contexte**

---

## 1. Contexte général

La **Question–Réponse contextuelle (Contextual Question Answering – QA)** est une tâche avancée du Traitement Automatique du Langage Naturel (NLP). Elle consiste à répondre automatiquement à une **question en langage naturel** en s’appuyant explicitement sur un **texte de contexte fourni**, dans lequel se trouve l’information recherchée.

Dans le cas de la **Darija marocaine**, cette tâche est particulièrement complexe en raison de :
- l’absence de corpus QA annotés ;
- la variabilité orthographique et syntaxique ;
- le code-switching (Darija / français / arabe standard) ;
- la rareté de modèles pré-entraînés adaptés.

Ce projet vise à confronter les étudiants à un **cas réel de QA pour langue low-resource**, en mettant l’accent sur :
- la **constitution d’un corpus QA contextuel annoté** ;
- l’**annotation semi-automatique assistée par LLM** ;
- la comparaison de différentes approches de Question–Réponse.

---

## 2. Objectifs du projet

### 2.1 Objectifs pédagogiques

À l’issue du projet, l’étudiant devra être capable de :
- collecter des textes de contexte en Darija ;
- formuler et structurer des paires **question–réponse** ;
- concevoir une pipeline NLP dédiée à la QA contextuelle ;
- mettre en œuvre une **annotation semi-automatique de données QA** ;
- entraîner et comparer plusieurs approches de Question–Réponse ;
- analyser de manière critique les performances et erreurs du système.

### 2.2 Compétences visées

- NLP appliqué aux langues peu dotées  
- Question–Réponse contextuelle  
- Data annotation structurée (texte, question, réponse)  
- Machine Learning / LLM  
- Reproductibilité et suivi expérimental  

---

## 3. Périmètre et contraintes

- Projet réalisé en **binôme**
- Langage principal : **Python**
- Langue cible : **Darija marocaine**
- Données issues **exclusivement de sources publiques**
- Utilisation obligatoire :
  - d’une **annotation semi-automatique (LLM + validation humaine)** ;
  - d’un outil de **tracking expérimental** (MLflow ou Weights & Biases).
- Rapport technique et code documenté exigés.

---

## 4. Définition de la tâche de Question–Réponse contextuelle

### 4.1 Nature de la tâche

La tâche consiste à développer un système capable de répondre à une **question en Darija** en s’appuyant sur un **texte de contexte en Darija**, fourni explicitement en entrée.

Chaque instance du dataset est structurée sous la forme :
(contexte, question) → réponse
La réponse doit :
- être **contenue ou inférable à partir du contexte** ;
- être concise et pertinente ;
- ne pas introduire d’informations externes au texte.

---

### 4.2 Type de QA retenu

Une instance QA est définie comme :

> Une **question claire et non ambiguë**, formulée en Darija,  
> dont la **réponse est une sous-chaîne exacte du contexte**.

#### Contraintes absolues

- ❌ aucune réponse générée librement
- ❌ aucune réponse paraphrasée
- ❌ aucune réponse issue de connaissances externes
- ✅ la réponse doit être **copiée mot pour mot du contexte**

### 4.3 Typologie des questions autorisées

Les questions doivent porter sur des **faits explicitement mentionnés** dans le premier paragraphe :

### Types recommandés
- Qui ? (personne, institution)
- Où ? (lieu précis)
- Quand ? (date, période)
- Quoi ? (événement, décision, action)
- Combien ? (chiffres, quantités)
- Quel / Quelle ? (entité unique)

### Types interdits
- Pourquoi ?
- Comment ?
- Questions implicites ou interprétatives
- Jugements de valeur
- Questions nécessitant inférence externe

---

### 4.4 Unité de traitement

- Contexte : paragraphes ou articles courts (100 à 500 mots)
- Question : phrase interrogative en Darija
- Réponse : mot, groupe de mots ou phrase courte

---

### 4.5 Contraintes linguistiques spécifiques à la Darija

Les systèmes développés devront explicitement prendre en compte :
- la variation orthographique ;
- le code-switching Darija / FR / AR ;
- l’usage de l’alphabet arabe ou latin (Arabizi) ;
- les formulations orales et elliptiques des questions.

Ces aspects devront être analysés dans l’évaluation des erreurs.

---

## 5. Sources de données et constitution du corpus QA

Les données sources sont déjà scrapées et fournies aux étudiants sous forme de fichiers complémentaires.
Aucune nouvelle collecte n’est requise.

### 5.1 Articles Goud.ma

- **Source** : https://www.goud.ma
- **Période couverte** : septembre 2024 → janvier 2026
- **Langue** : Darija marocaine (arabe dialectal principalement)
- **Type de contenu** :
  - actualité nationale,
  - faits divers,
  - politique,
  - société,
  - économie,
  - sport,
  - culture.

Les articles sont considérés comme des **documents d’information**, adaptés à la QA contextuelle grâce à leur structure narrative et factuelle.

---

## 6. Étape 1 – Prétraitement et exploration

### 6.1 Principe fondamental

Pour chaque article Goud.ma :

> **Seul le premier paragraphe de l’article est utilisé comme contexte.**

### Justification
- Le premier paragraphe contient généralement :
  - le résumé des faits essentiels,
  - les entités clés (personnes, lieux, dates),
  - l’information à plus forte densité sémantique.
- Cela réduit :
  - la longueur du contexte,
  - les ambiguïtés,
  - le nombre de réponses possibles.

### 6.2 Définition du champ `context`

Le champ `context` correspond exactement :
- au **premier paragraphe brut**,
- sans reformulation,
- sans normalisation lourde,
- sans suppression d’informations.

### 6.3 Prétraitement

- nettoyage léger (URLs, emojis, spam) ;
- normalisation minimale et justifiée ;
- segmentation claire du contexte.

### 6.4 Analyse exploratoire

- longueur des contextes ;
- types de questions (qui, quoi, où, pourquoi, comment) ;
- distribution des réponses ;
- analyse du code-switching.

---

## 7. Étape 2 – Annotation semi-automatique QA (étape centrale)

### 7.1 Objectif

Construire un corpus QA fiable en combinant :
- génération automatique de questions et réponses par LLM ;
- validation et correction humaines ciblées ;
- contrôle qualité systématique.

---

### 7.2 Démarche recommandée d’annotation

#### Étape 1 – Jeu de référence annoté manuellement

- Sélectionner **800 à 1 500 contextes**.
- Rédiger manuellement les questions et réponses.
- Définir un **guide d’annotation QA**.
- Constituer un **gold standard partiel**.

---

#### Étape 2 – Génération automatique par LLM

##### Modèle recommandé

* **Gemini 2.5** ou **Gemini 3**
* Accès via :

  * API Google
  * Google Colab

---

##### Prompt (extractive QA)

Les étudiants doivent utiliser un prompt **strictement contraint**, par exemple :

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

#### Étape 3 – Validation assistée avec Label Studio

Les triplets (contexte, question, réponse) doivent être importés dans **Label Studio**.

Les annotateurs humains doivent :
- valider la pertinence de la question ;
- vérifier l’exactitude de la réponse ;
- corriger ou rejeter les triplets incorrects.

---

#### Étape 4 – Contrôle qualité

- taux de correction des paires générées ;
- analyse des erreurs fréquentes (questions ambiguës, réponses hors contexte) ;
- homogénéisation du corpus final.

⚠️ **La validation humaine est impérative et non optionnelle.**

Les annotateurs doivent :

- Supprimer
  * questions ambiguës,
  * réponses apparaissant plusieurs fois dans le contexte,
  * questions dont la réponse nécessite interprétation.
- Corriger
  * erreurs linguistiques légères dans la question,
  * mauvaise sélection du span de réponse.
- Vérifier systématiquement
  * unicité de la réponse,
  * correspondance exacte caractère par caractère,
  * cohérence question ↔ réponse.

##### Exemple correct

**Contexte**
```

قالت مصادر مطلعة أن المحكمة الابتدائية بالدار البيضاء قررت تأجيل
محاكمة المتهم إلى غاية 15 أكتوبر 2024.

````

**QA**
```json
{
  "context": "قالت مصادر مطلعة أن المحكمة الابتدائية بالدار البيضاء قررت تأجيل محاكمة المتهم إلى غاية 15 أكتوبر 2024.",
  "question": "إلى غاية شحال تأجلات المحاكمة؟",
  "answer": "15 أكتوبر 2024"
}
````

---

### Exemple incorrect (réponse non verbatim)

```json
{
  "question": "فين كانت المحكمة؟",
  "answer": "في الدار البيضاء"
}
```

❌ La réponse **n’est pas une sous-chaîne exacte** du contexte.

---

##### Exemple incorrect (ambiguïté)

```json
{
  "question": "شكون المعني بالقضية؟",
  "answer": "المتهم"
}
```

❌ Plusieurs entités pourraient correspondre.

#### Critères de qualité du corpus

Le corpus final doit satisfaire :

* 100 % des réponses sont extractives ;
* aucune réponse n’est générée librement ;
* chaque question admet **une seule réponse correcte** ;
* le contexte est suffisant pour répondre sans connaissance externe.

---

---

## 8. Étape 3 – Modélisation de la QA contextuelle

L’objectif est de comparer la capacité de différentes architectures à identifier avec précision le "span" (segment de texte) contenant la réponse au sein d'un contexte en Darija.

### 8.1 Approches attendues

#### A. Baseline : Machine Learning classique

Avant d'utiliser des modèles lourds, les étudiants doivent établir un point de comparaison avec des méthodes de recherche d'information traditionnelles :

* **Similarité de termes (TF-IDF / BM25) :** Calcul de la pertinence entre la question et les phrases du contexte pour extraire la phrase la plus probable.
* **Embeddings statiques :** Utilisation de vecteurs de mots (type FastText pré-entraîné sur des données maghrébines) pour calculer une similarité cosinus moyenne entre la question et le contexte.

#### B. QA supervisée extractive (Transformers)

C’est l’approche privilégiée pour le NLP moderne. Elle consiste à prédire les indices de début et de fin de la réponse.

* **DarijaBERT :** Utilisation de modèles de langue spécialisés comme **DarijaBERT** ou **MarBERT**. Ces modèles, pré-entraînés sur des milliards de tokens de dialectes arabes, sont particulièrement robustes face aux spécificités de la Darija.
* **Gestion du multi-script :** Évaluation de la capacité du modèle à traiter à la fois l'alphabet arabe et l'**Arabizi** (caractères latins), très présent dans les données Reddit et YouTube collectées.

---

### 8.2 Prétraitement spécifique à la modélisation

* **Tokenisation :** Adaptation du tokenizer pour ne pas briser les structures morphologiques complexes de la Darija.
* **Normalisation :** Choix stratégique (ou non) de normaliser certains caractères (ex: 3, 7, 9 en Arabizi) avant l'injection dans les modèles BERT.

---

### 8.4 Évaluation

- Exact Match (EM)
- F1-score
- Analyse qualitative des réponses incorrectes:
  * **Erreurs de frontières :** Le modèle trouve la réponse mais inclut des mots superflus avant ou après.
  * **Erreurs de type :** Le modèle donne un nom alors qu'on attend un chiffre (ex: "شحال" -> "بزاف" au lieu de "10").
  * **Impact du code-switching :** Analyse de la performance selon que la question/réponse est en caractères arabes ou en Arabizi.
  * **Hallucinations (si approche générative testée) :** Cas où la réponse n'est pas présente dans le contexte fourni.

---

## 9. Étape 4 – Suivi expérimental

Toutes les expérimentations doivent être tracées à l’aide de :
- **MLflow**, ou
- **Weights & Biases**.

Chaque run doit inclure :
- version du corpus ;
- type de modèle QA ;
- paramètres ;
- métriques obtenues.

---

## 10. Livrables attendus

1. Corpus QA contextuel annoté en Darija  
2. Rapport technique détaillé  
3. Code source structuré et documenté  
4. Dashboard de tracking expérimental  
5. Analyse qualitative des erreurs QA  

---

## 11. Critères d’évaluation (indicatif)

| Critère | Pondération |
|------|-------------|
| Qualité du corpus et annotation | 50 % |
| Pipeline NLP / QA | 15 % |
| Modélisation et évaluation | 10 % |
| Tracking et reproductibilité | 10 % |
| Analyse critique et rapport| 15 % |

---

## 12. Extensions possibles (bonus)

- QA multi-documents
- Comparaison extractif vs génératif
- Étude de l’impact de la longueur du contexte
- Intégration QA + chatbot Darija

---
