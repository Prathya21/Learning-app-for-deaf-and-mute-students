# EduSign Dataset Inspection Report: `vidit031/isl-isolated-40words`

**Date**: 2025-09-03  
**Phase**: 5B — Dataset Acquisition & Inspection  
**Status**: **CRITICAL DOWNLOAD FAILURE — NO VALID VIDEOS**

---

## 1. ACTUAL DOWNLOADED DATASET STATISTICS

| Metric | Value |
|--------|-------|
| **Total metadata entries** | 642 |
| **Files downloaded** | 456 (attempted) |
| **Valid video files** | **0** |
| **Corrupted/error files** | 456 (all HTML 429 error pages) |
| **Usable video data** | **0 seconds** |

**Root Cause**: The ISL500/ISL-DATA source (HuggingFace dataset `ISL500/ISL-DATA`) rate-limited all requests (HTTP 429 Too Many Requests). The download script saved the HTML error response pages as `.mp4` files. **Zero valid MP4 video files were obtained.**

---

## 2. METADATA-BASED CLASS DISTRIBUTION (from HuggingFace metadata only)

*Based on HuggingFace metadata — not verified against actual video files since none exist.*

| Class | Metadata Samples | Sources (metadata) | Signers (metadata) | Status |
|-------|------------------|--------------------|--------------------|--------|
| hello | 29 | INCLUDE, ISL500 | 29 | Metadata only |
| thank you | 32 | INCLUDE, ISL500, CISLR | 30 | Metadata only |
| friend | 31 | INCLUDE, ISL500, CISLR | 29 | Metadata only |
| market | 29 | INCLUDE, ISL500, CISLR | 29 | Metadata only |
| school | 30 | INCLUDE, ISL500, CISLR | 29 | Metadata only |
| hospital | 21 | INCLUDE, CISLR | 21 | Metadata only |
| okay | 23 | INCLUDE, ISLRTC, CISLR | 23 | Metadata only |
| hello | 29 | INCLUDE, ISL500 | 29 | Metadata only |
| teacher | 10 | ISL500, CISLR | 9 | Metadata only |
| student | 10 | ISL500, CISLR | 9 | Metadata only |
| water | 11 | ISL500, CISLR | 9 | Metadata only |
| eat | 12 | ISL500, CISLR | 10 | Metadata only |
| drink | 10 | ISL500, CISLR | 10 | Metadata only |
| father | 10 | ISL500, CISLR | 10 | Metadata only |
| food | 10 | ISL500, CISLR | 9 | Metadata only |
| go | 10 | ISL500, CISLR | 9 | Metadata only |
| he | 10 | ISL500, CISLR | 9 | Metadata only |
| she | 10 | ISL500, CISLR | 9 | Metadata only |
| sister | 10 | ISL500, CISLR | 9 | Metadata only |
| sit | 12 | ISL500, CISLR | 9 | Metadata only |
| tea | 10 | ISL500, CISLR | 9 | Metadata only |
| teacher | 10 | ISL500, CISLR | 9 | Metadata only |
| today | 10 | ISL500, CISLR | 9 | Metadata only |
| water | 11 | ISL500, CISLR | 9 | Metadata only |
| what | 10 | ISL500, CISLR | 9 | Metadata only |
| where | 10 | ISL500, CISLR | 9 | Metadata only |
| yes | 10 | ISL500, CISLR | 9 | Metadata only |
| you | 11 | ISL500, CISLR | 9 | Metadata only |
| please | 11 | ISL500, CISLR | 9 | Metadata only |
| mother | 11 | ISL500, CISLR | 9 | Metadata only |
| no | 10 | ISL500, CISLR | 9 | Metadata only |
| help | 9 | ISL500, CISLR | 9 | Metadata only |
| sorry | 6 | ISLRTC, CISLR | 2 | Metadata only |
| come | 4 | ISLRTC, CISLR | 2 | Metadata only |
| stand | 2 | ISLRTC, CISLR | 2 | Metadata only |
| when | 2 | ISLRTC, CISLR | 2 | Metadata only |
| read | 3 | ISLRTC, CISLR | 2 | Metadata only |
| write | 3 | ISLRTC, CISLR | 2 | Metadata only |
| brother | 3 | CISLR | 1 | Metadata only |
| goodbye | 3 | ISLRTC, CISLR | 2 | Metadata only |
| me | 3 | ISLRTC, CISLR | 2 | Metadata only |
| stop | 4 | ISLRTC, CISLR | 2 | Metadata only |
| home | 1 | CISLR | 1 | Metadata only |
| stand | 2 | ISLRTC, CISLR | 2 | Metadata only |
| when | 2 | ISLRTC, CISLR | 2 | Metadata only |

---

## 3. RECOMMENDED INITIAL CLASS SUBSET

**BASED ON METADATA ONLY — UNVERIFIED AGAINST ACTUAL VIDEOS**

| Class | Metadata Samples | Sources | Signers | Verdict |
|-------|------------------|---------|---------|---------|
| hello | 29 | INCLUDE, ISL500 | 29 | Would keep |
| thank you | 32 | INCLUDE, ISL500, CISLR | 30 | Would keep |
| friend | 31 | INCLUDE, ISL500, CISLR | 29 | Would keep |
| market | 29 | INCLUDE, ISL500, CISLR | 29 | Would keep |
| school | 30 | INCLUDE, ISL500, CISLR | 29 | Would keep |
| hospital | 21 | INCLUDE, CISLR | 21 | Would keep |
| okay | 23 | INCLUDE, ISLRTC, CISLR | 23 | Would keep |
| teacher | 10 | ISL500, CISLR | 9 | Would keep |
| student | 10 | ISL500, CISLR | 9 | Would keep |
| water | 11 | ISL500, CISLR | 9 | Would keep |
| eat | 12 | ISL500, CISLR | 10 | Would keep |
| drink | 10 | ISL500, CISLR | 10 | Would keep |
| father | 10 | ISL500, CISLR | 10 | Would keep |
| food | 10 | ISL500, CISLR | 9 | Would keep |
| go | 10 | ISL500, CISLR | 9 | Would keep |
| he | 10 | ISL500, CISLR | 9 | Would keep |
| she | 10 | ISL500, CISLR | 9 | Would keep |
| sister | 10 | ISL500, CISLR | 9 | Would keep |
| sit | 12 | ISL500, CISLR | 9 | Would keep |
| tea | 10 | ISL500, CISLR | 9 | Would keep |
| today | 10 | ISL500, CISLR | 9 | Would keep |
| water | 11 | ISL500, CISLR | 9 | Would keep |
| what | 10 | ISL500, CISLR | 9 | Would keep |
| where | 10 | ISL500, CISLR | 9 | Would keep |
| yes | 10 | ISL500, CISLR | 9 | Would keep |
| you | 11 | ISL500, CISLR | 9 | Would keep |
| please | 11 | ISL500, CISLR | 9 | Would keep |
| mother | 11 | ISL500, CISLR | 9 | Would keep |
| no | 10 | ISL500, CISLR | 9 | Would keep |
| help | 9 | ISL500, CISLR | 9 | Would keep |
| sorry | 6 | ISLRTC, CISLR | 2 | **Borderline** |
| **Total classes ≥5 samples**: 30 | | | |

**Classes to exclude (<5 samples)**: brother(3), come(4), goodbye(3), me(3), read(3), write(3), stand(2), when(2), home(1)

---

## 4. SIGNER-INDEPENDENT SPLIT FEASIBILITY

### Signer Distribution (metadata)
- **15 ISL500 users** (`User001`–`User015`): 27 samples each, 27 classes each (full coverage of their 27 classes)
- **User009**: 3 samples, 3 classes
- **User010–User015**: 0 samples in downloaded metadata
- **CISLR signers** (`action`, `relation`, `abstract`, etc.): 1–16 samples, narrow class coverage
- **INCLUDE signers** (`include_MVI_XXXX`): mostly 1 sample each
- **ISLRTC signers** (`islrtc_dict`, `greeting`, etc.): 1–13 samples

### Critical Finding: Signer-independent split NOT FEASIBLE for 30-class set

| Missing Classes if 3 ISL500 users held out | Source |
|--------------------------------------------|--------|
| sorry | CISLR/ISLRTC only |
| okay | INCLUDE/ISLRTC/CISLR |
| hospital | INCLUDE/CISLR only |

**Analysis**: The classes `sorry`, `okay`, `hospital` are **only present in non-ISL500 sources** (CISLR, INCLUDE, ISLRTC). Since ISL500 users only cover 27/30 classes, holding out any 3 ISL500 users still leaves all 27 ISL500 classes covered by remaining 12 users, but the 3 non-ISL500 classes would have zero coverage in training.

**Conclusion**: **True signer-independent split NOT FEASIBLE for full 30-class set** without cross-source contamination. The dataset's multi-source nature breaks signer independence.

---

## 5. MEDIAPIPE CPU SMOKE TEST

**RESULT: NOT PERFORMABLE — NO VALID VIDEO FILES EXIST**

| Test | Status |
|------|--------|
| Download valid videos | ❌ FAILED — 456 HTML error pages |
| MediaPipe Hands on CPU | ⚠️ SKIPPED — No valid input |
| MediaPipe HandLandmarker | ⚠️ SKIPPED — No valid input |

**Note**: MediaPipe Tasks Vision v1.0.1 crashes on Apple Silicon (Metal GPU delegate crash) even with CPU delegate requested. This is a known environment issue. Would require CPU-only fallback or workaround.

---

## 6. VIDEO DURATION/FRAME STATISTICS

**UNAVAILABLE — NO VALID VIDEO FILES TO ANALYZE**

*Metadata reports*:
- Duration range: 1.73s – 14.27s (median 3.23s per metadata)
- FPS: 30 FPS (all samples per metadata)
- Resolution: 854×480 (82%), 480×480 (18%)

**Recommended T (temporal frames) if videos existed**:
- At 30 FPS: 32 frames = ~1.07s, 64 frames = ~2.13s
- Median duration 3.23s ≈ 97 frames at 30 FPS
- **Recommendation**: T=64 (covers ~90% of videos) with padding/truncation

---

## 7. LICENSE / PROVENANCE RISKS

### Source Attribution Required
| Upstream Source | Samples (metadata) | License | Attribution Required |
|-----------------|-------------------|---------|---------------------|
| INCLUDE (AI4Bharat) | 143 | CC-BY-4.0 | ✅ Yes |
| CISLR (Exploration-Lab) | 81 | AFL-3.0 | ✅ Yes |
| ISL500 / ISL-DATA | 219 | "research (see HF card)" | ⚠️ Ambiguous |
| ISLRTC Dictionary | 13 | Government open data / MIT | ✅ Yes |

### Risk Assessment
- **Mixed licenses** — No single license covers entire dataset
- **ISL500 license ambiguous** — "research (see HF card)" is not a standard license
- **Public demo requires** attribution to all 4 sources
- **Commercial use** — AFL-3.0 and ambiguous ISL500 license may restrict

---

## 8. RECOMMENDED PREPROCESSING REPRESENTATION (IF VIDEOS EXISTED)

```
Video (MP4, 30 FPS)
    ↓
MediaPipe HandLandmarker (CPU, VIDEO mode)
    ├── 21 landmarks × up to 2 hands
    ├── x, y, z normalized [0,1]
    └── handedness + confidence
    ↓
Per-frame: [hand1[21,3], hand2[21,3]] → 126-dim vector
    ↓
Wrist-centric normalization (translate wrist→origin)
    ↓
Scale normalization (divide by hand bbox diagonal)
    ↓
Uniform temporal sampling → Fixed T frames
    ↓
Padding (zero) / truncation to T
    ↓
Tensor: [Batch, T, 126]
```

**Extensibility for future hands+pose**:
- Schema: `[hand1[21,3], hand2[21,3], pose[33,3]?]`
- Pose landmarks optional (add later if needed)
- No face landmarks (grammatical non-manual markers — future phase)

---

## 9. FINAL DECISION: **NO-GO**

### Why NO-GO:

| Blocker | Severity | Details |
|---------|----------|---------|
| **Zero valid videos** | 🔴 CRITICAL | All 456 downloads are HTML 429 error pages |
| **ISL500 source completely failed** | 🔴 CRITICAL | 219 metadata samples (34%) from ISL500 — all failed |
| **No signer-independent split** | 🟠 HIGH | 3 classes only in non-ISL500 sources |
| **Mixed/ambiguous licenses** | 🟡 MEDIUM | ISL500 license unclear; AFL-3.0 for CISLR |
| **MediaPipe crashes on Apple Silicon** | 🟡 MEDIUM | GPU delegate crash even with CPU request |

---

## 10. RECOMMENDED ALTERNATIVE PATH

### Option A: Use INCLUDE-50 directly from Zenodo
- **Size**: ~54 GB download (all categories) — exceeds 22 GB disk
- **Workaround**: Selective category download (Greetings + People + Home ≈ 9 GB)
- **Classes available**: HELLO, THANK_YOU, GOOD_MORNING, TEACHER, STUDENT, FATHER, MOTHER, BROTHER, SISTER, etc.

### Option B: Find alternative ISL dataset
- **ISL-CSLRT** (sentence-level, 100 sentences)
- **WLASL** (ASL, 2000 classes) — for transfer learning
- **Custom recording** — Record 20-30 signs with 2-3 signers locally

### Option C: Synthetic/Procedural approach
- Use MediaPipe to generate synthetic landmark sequences from existing gloss definitions
- Train lightweight classifier on synthetic data
- Validate with small real-world recordings

---

## 11. NEXT STEPS (IF APPROVED)

1. **Pivot to Option A**: Selective INCLUDE-50 download (Greetings + People categories ≈ 9 GB)
2. **Verify actual videos** from Zenodo before full download
3. **Build preprocessing pipeline** on verified videos
4. **Train 15-20 class model** with signer-independent split using ISL500 users
5. **Integrate with EduSign** gesture-to-text module

---

## APPROVAL REQUESTED

**Do not proceed with current dataset.** The `vidit031/isl-isolated-40words` download has **completely failed** — zero usable videos.

**Please approve one of:**
- **Option A**: Pivot to INCLUDE-50 (selective Zenodo download)
- **Option B**: Search for alternative ISL dataset with working downloads
- **Option C**: Record custom dataset locally (20 signs × 3 signers × 5 reps = 300 videos)

**Awaiting approval before any further dataset work.**