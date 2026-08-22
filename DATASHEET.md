# Datasheet for BangladeshiVQA

Following the *Datasheets for Datasets* framework (Gebru et al.). This mirrors the paper's
Ethics and Broader Impact section and documents how the dataset was created, what it
contains, and how it should be used.

## Motivation

**For what purpose was the dataset created?**
To provide a **culturally grounded, native Bangla** Visual Question Answering benchmark.
Existing multilingual VQA resources rely on Western-centric imagery or machine translation,
filtering out local visual semantics (festival attire, regional transport, cuisine, Bangla
signage). BangladeshiVQA measures whether vision-language models can perceive **and reason
about** Bangladeshi visual context — including the **first Bangla in-image scene-text (OCR)
questions** in a Bangla VQA benchmark.

**Who created it?**
The dataset authors (see [CITATION.cff](CITATION.cff)); annotation was performed by two
expert native-Bangla reviewers with academic backgrounds in multimodal AI.

## Composition

- **Instances:** 2,068 images and 7,038 open-ended Bangla question–answer pairs
  (mean 3.40 QA/image).
- **What each instance represents:** a QA record — `image_id`, `q_type`, `question`,
  `answer`, `level` — associated with one image.
- **Labels:** question type ∈ {Existence, Counting, Attribute, Reasoning, OCR}; cognitive
  level ∈ {Level 1 Perception, Level 2 Representation, Level 3 Cognition}, derived from type.
- **Splits:** train 5,628 / val 704 / test 706, split **strictly by image ID** so no image
  appears in more than one split (prevents train/test leakage).
- **Language:** Bangla (Bengali). Answers average 1.44 words; questions 5.87 words.
- **What is NOT included:** no names, identity labels, biometric annotations, or
  person-recognition tasks. Per-image captions and category metadata used internally during
  construction are **not released**; the benchmark ships as QA pairs + splits (category
  distribution is reported only in aggregate).

## Collection process

Images were collected from three sources (2,068 total):

| Source | Count | Notes |
|---|---:|---|
| Primary crowdsourcing | 1,170 | Photographed by authors + local volunteers in Bangladesh |
| Instagram public posts | 705 | Curated from ≈30,000 via cultural hashtags, then filtered (~2.35% retained) |
| Bangladeshi news portals | 193 | Public news pages |

Instagram images were filtered in four phases: (1) automated low-resolution removal,
(2) perceptual-hash de-duplication, (3) CLIP ViT-B/32 relevance filtering against
Bangladeshi context, (4) manual human review removing private/explicit/sensitive images.
All images were symmetrically padded and resized to 1024×1024 to keep Bangla scene text
readable, and renamed to `CS_xxxx`.

## Annotation process

A human-in-the-loop pipeline with **Gemini 3.5 Flash** and two expert reviewers:

1. Gemini generated an initial objective Bangla caption per image; reviewers corrected
   inaccuracies, removed hallucinations, and added cultural context.
2. Gemini drafted QA pairs from the validated captions (**10,186 raw pairs**).
3. Reviewers edited, filtered, re-labelled question types, and added missing culturally
   grounded questions, checking every pair against the actual image → **7,038 final pairs**
   (3,148 fewer than the raw draft).

**Inter-rater agreement:** on 763 QA pairs from 200 randomly sampled images, keep/discard
decisions reached **Cohen's κ = 0.83**; disagreements were resolved by discussion.

## Ethics: consent, minors, and sensitive content

- **Consent:** informed consent was obtained from identifiable subjects in the 1,170
  primary crowdsourced images, for non-commercial academic research use. For public
  Instagram/news images, individual consent was not obtainable; these were manually
  screened, and private/explicit/vulnerable/sensitive scenes were removed.
- **Minors:** images with minors were retained only for ordinary, non-sensitive activities
  (e.g., studying, playing); images of minors in vulnerable/private/sensitive settings were
  excluded. No task identifies, categorizes, or tracks minors.
- **Sensitive content:** culturally and religiously sensitive images were manually reviewed.
- **Privacy:** EXIF/GPS metadata, usernames, captions, comments, and platform identifiers
  were removed; all 2,068 released images were verified with ExifTool to contain no EXIF/GPS
  metadata; files were renamed to prevent recovery of source usernames/filenames.

## Preprocessing / cleaning

Bangla spelling normalization (Bangla Academy rules), numeral/punctuation normalization for
scoring, de-duplication, low-resolution/irrelevant filtering, EXIF stripping, square
padding + resize. These steps are described in the paper; the released code covers caption/QA
generation, evaluation, and fine-tuning (see [`code/`](code/)).

## Uses

- **Intended use:** non-commercial academic research — evaluating and fine-tuning
  vision-language models on Bangladeshi geo-cultural and tiered multimodal reasoning.
- **Out-of-scope / discouraged uses:** commercial use; any person identification,
  biometric, or surveillance application; treating the dataset as representative of all of
  Bangladesh (it is a curated cultural sample). BERTScore should be read as a semantic-
  relevance signal, not a correctness measure.

## Distribution

- **Code, fine-tuning config + LoRA adapter, and a preview sample:** this GitHub
  repository, under [CC BY-NC 4.0](LICENSE).
- **Full dataset — all 2,068 images plus the 7,038 QA pairs and train/val/test splits:**
  hosted on Hugging Face, with an optional citable archival snapshot on Zenodo (see README
  [Download](README.md#download)). Annotations are released under [CC BY-NC 4.0](LICENSE);
  images remain under original ownership for non-commercial academic use ([NOTICE.md](NOTICE.md)).

## Maintenance

- **Maintainer / contact:** tanim494@gmail.com.
- **Takedown:** any depicted person or rights holder may request image removal; see
  [TAKEDOWN.md](TAKEDOWN.md). Confirmed images are removed from future versions.
- **Errata / updates:** tracked via the GitHub repository; the Zenodo record provides
  citable, versioned archival snapshots.
