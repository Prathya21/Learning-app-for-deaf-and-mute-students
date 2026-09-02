# INCLUDE Dataset Acquisition Report

**Date**: 2025-09-03  
**Phase**: 5B — INCLUDE Subset Acquisition  
**Status**: **DOWNLOAD BLOCKED — ZENODO BANDWIDTH LIMITATION**

---

## Executive Summary

Attempted selective download of INCLUDE-50 category archives from the authoritative Zenodo source (`https://zenodo.org/records/4010759`). The download is **blocked by severe bandwidth limitations** (~200-300 KB/s from Zenodo), making full acquisition impractical within session time limits.

**No valid video files were acquired.** The partial download of `Greetings_1of2.zip` (1.4 GB / 1.5 GB) was incomplete and corrupt.

---

## Target Subset Plan

Based on INCLUDE-50 vocabulary analysis, the following categories were prioritized for EduSign:

| Category | INCLUDE-50 Classes | Archives | Zenodo Size | Priority |
|----------|-------------------|----------|-------------|----------|
| **Greetings** | Hello, Thank you, Good Morning | 2 (1of2, 2of2) | 2.65 GB | 🔴 Critical |
| **People** | Father, Brother, Boy, Girl | 5 (1of5-5of5) | 6.45 GB | 🔴 Critical |
| **Jobs** | Teacher | 2 (1of2, 2of2) | 3.01 GB | 🟠 High |
| **Places** | House, Court, Store/Shop, Bank | 4 (1of4-4of4) | 5.03 GB | 🟠 High |
| **Animals** | Dog, Bird, Cow | 2 (1of2, 2of2) | 2.75 GB | 🟢 Medium |
| **Colours** | Red, Black, White | 2 (1of2, 2of2) | 2.59 GB | 🟢 Medium |

**Total for full overlap**: ~22.5 GB download → ~45-60 GB extracted  
**Available disk**: 22 GB → **EXCEEDS CAPACITY**

### Recommended Minimal Viable Subset (fits 22 GB disk)

| Category | Archives | Download | Extracted (est.) | Classes |
|----------|----------|----------|------------------|---------|
| Greetings | 2 | 2.65 GB | 5-8 GB | Hello, Thank you, Good Morning |
| People (1of5 only) | 1 | 1.27 GB | 2-3 GB | Subset of Father/Brother/etc. |
| Jobs | 2 | 3.01 GB | 6-9 GB | Teacher |
| **Total** | **5** | **~6.9 GB** | **~13-20 GB** | **8 classes** |

---

## Download Attempt Log

### Attempt 1: Greetings_1of2.zip (1.5 GB)
- **Source**: `https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content`
- **Speed**: ~200-300 KB/s (highly variable)
- **Time for 1.5 GB**: ~1.5-2 hours
- **Result**: Download stalled at 1.4 GB (93%), zip file corrupt/incomplete
- **Cleanup**: Removed incomplete file

### Attempt 2: Resume with curl -C -
- **Speed**: Still ~200-300 KB/s
- **ETA**: ~1-2 hours remaining
- **Session timeout**: Command exceeded 120s timeout repeatedly

---

## Root Cause Analysis

| Factor | Impact |
|--------|--------|
| **Zenodo bandwidth** | ~200-300 KB/s sustained (no CDN acceleration) |
| **File sizes** | 1-1.7 GB per archive |
| **Total archives needed** | 10-15 for full subset |
| **Available disk** | 22 GB (insufficient for extracted full subset) |
| **Session timeout** | 120s per command (insufficient for multi-hour downloads) |

---

## Alternative Acquisition Strategies

### Option A: HuggingFace Dataset (METADATA ONLY)
- **Dataset**: `ai4bharat/INCLUDE` on HuggingFace
- **Contains**: Full metadata (642 samples, labels, sources, signers, video paths)
- **Videos**: NOT hosted on HF — only references to Zenodo URLs
- **Speed**: Fast metadata access, but no video files

### Option B: Selective Zenodo Download (OFFLINE/ASYNC)
- **Approach**: Run downloads in background/overnight using `curl -C -` or `wget -c`
- **Tooling**: Use `aria2c` (16 connections) if available for parallel download
- **Estimated time**: 4-6 hours for minimal subset (6.9 GB)

### Option C: Alternative Source — ISL-Isolated-40words
- **Dataset**: `vidit031/isl-isolated-40words` (HuggingFace)
- **Size**: ~32 MB total (642 videos)
- **Overlap**: 26/33 EduSign classes (79%)
- **Status**: **PROVEN DOWNLOAD FAILURE** — ISL500 source rate-limits (HTTP 429)

### Option D: Custom Local Recording
- **Scope**: 20 signs × 3 signers × 5 reps = 300 videos
- **Time**: ~2-3 hours recording + processing
- **Advantage**: Full control, perfect vocabulary match, signer-independent design
- **Cost**: Requires human signers + camera setup

### Option E: Synthetic Landmark Generation
- **Approach**: Generate synthetic landmark sequences from gloss definitions
- **Use case**: Prototype classifier development without real videos
- **Limitation**: No visual validation, synthetic bias

---

## Recommended Path Forward

### Immediate (Session-Compatible): Option D + E Hybrid
1. **Create synthetic landmark dataset** for all 33 EduSign classes
2. **Use for classifier prototyping** (BiLSTM/TCN architecture)
3. **Defer real video acquisition** to dedicated session

### Next Session: Option B (Background Download)
```bash
# Install aria2c for parallel download
brew install aria2

# Download minimal subset in background
aria2c -x 16 -s 16 -d backend/data/include_subset/videos \
  "https://zenodo.org/api/records/4010759/files/Greetings_1of2.zip/content" \
  "https://zenodo.org/api/records/4010759/files/Greetings_2of2.zip/content" \
  "https://zenodo.org/api/records/4010759/files/People_1of5.zip/content" \
  "https://zenodo.org/api/records/4010759/files/Jobs_1of2.zip/content" \
  "https://zenodo.org/api/records/4010759/files/Jobs_2of2.zip/content"
```

---

## Vocabulary Coverage Summary

| EduSign Gloss | INCLUDE-50 | Available in Planned Subset |
|---------------|------------|----------------------------|
| HELLO | ✅ Greetings | ✅ Yes |
| THANK_YOU | ✅ Greetings | ✅ Yes |
| PLEASE | ❌ | ❌ No |
| YES | ❌ | ❌ No |
| NO | ❌ | ❌ No |
| STUDENT | ❌ | ❌ No |
| TEACHER | ✅ Jobs | ✅ Yes |
| LEARN | ❌ | ❌ No |
| BOOK | ❌ | ❌ No |
| WATER | ❌ | ❌ No |
| GOOD | ✅ Adjectives | ❌ No (Adjectives not in minimal subset) |
| BAD | ❌ | ❌ No |
| HELP | ❌ | ❌ No |
| UNDERSTAND | ❌ | ❌ No |
| QUESTION | ❌ | ❌ No |
| READ | ❌ | ❌ No |
| WRITE | ❌ | ❌ No |
| WHAT | ❌ | ❌ No |
| WHERE | ❌ | ❌ No |
| HOW | ❌ | ❌ No |

**Overlap with minimal subset**: 3/15 (20%) — only Hello, Thank you, Teacher

---

## Disk Space Reality Check

| Item | Size |
|------|------|
| Available | 22 GB |
| Greetings (2 zips) | 2.65 GB |
| People_1of5 | 1.27 GB |
| Jobs (2 zips) | 3.01 GB |
| **Total download** | **6.93 GB** |
| **Extracted (est. 2-3×)** | **14-21 GB** |
| **Remaining** | **1-8 GB** |
| **Margin** | **TIGHT** |

---

## Blocker Summary

| Blocker | Severity | Resolution |
|---------|----------|------------|
| Zenodo bandwidth | 🔴 Critical | Use aria2c parallel download; run overnight |
| Disk space | 🟠 High | Minimal subset only; monitor with `df -h` |
| Session timeout | 🟠 High | Use background tools (aria2c, screen, nohup) |
| Incomplete vocab | 🟡 Medium | Accept 3/15 overlap; supplement with synthetic |

---

## Immediate Action Items

1. **Install aria2c**: `brew install aria2`
2. **Start background download** of minimal subset (5 archives)
3. **Monitor disk**: `watch -n 60 df -h /Users/aswinimakam/signlingo`
4. **Verify each archive**: `unzip -t` after download
5. **Extract incrementally**: One archive at a time, verify, then delete zip

---

## Conclusion

**The INCLUDE dataset cannot be acquired within the current session due to Zenodo bandwidth limitations.** 

**Recommended**: 
1. Accept synthetic landmark dataset for immediate classifier prototyping (Phase 5C)
2. Schedule dedicated background download session for real videos
3. Proceed to Phase 5C (Preprocessing Pipeline Design) with synthetic data

**No valid videos acquired in this session.** The partial zip was removed.