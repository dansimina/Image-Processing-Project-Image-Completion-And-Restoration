# 🖼️ Patch-Based Image Inpainting

A C++ application built with OpenCV that automatically fills in missing or selected regions of an image using **patch-based reconstruction** techniques inspired by the PatchMatch algorithm. The user interactively selects a region to remove, and the algorithm seamlessly reconstructs it using information from the rest of the image.

---

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Algorithm Details](#algorithm-details)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

---

## Overview

This project implements **patch-based image inpainting** — a technique that reconstructs missing regions in an image by finding and copying visually similar patches from the known (unmasked) parts of the image. The algorithm prioritizes structural coherence and produces visually plausible results across a wide range of image types.

**Key Features:**
- Interactive mouse-based region selection
- Priority-driven patch filling (fills important structures first)
- PatchMatch-inspired propagation + random search for fast nearest-neighbor matching
- Gaussian post-processing for smooth transitions
- Automatic preprocessing to handle large images

---

## How It Works

The application follows this pipeline:

```
[Load Image] → [Preprocessing] → [Select Region] → [Generate Mask]
     → [Inpainting] → [Post-processing] → [Display Result]
```

1. **Load Image** — The user opens a `.bmp` image.
2. **Select Region** — The user draws a rectangle over the region to be removed using the mouse.
3. **Generate Mask** — A binary mask is created where `true` marks pixels to be filled.
4. **Inpainting** — The core algorithm fills the masked region patch by patch, using priority ordering and nearest-neighbor search.
5. **Post-processing** — A Gaussian filter smooths the transitions between filled and original regions.
6. **Display Result** — The completed image is shown side by side with the original.

---

## Algorithm Details

### Patch Representation
The image is processed in small square patches of size **7×7 pixels** (configurable via `PATCH_SIZE`). Each patch is identified by its center pixel coordinates.

### Priority Computation (`computePriority`)
Not all boundary patches are equal — the algorithm processes patches in order of importance. Priority is computed based on:
- **Confidence term** — how many known (unmasked) pixels are in the patch neighborhood
- **Data term** — presence of strong edges/gradients near the boundary
- **Color variance** — diversity of colors in the surrounding area

Patches with higher priority are filled first, ensuring that strong linear structures and edges are reconstructed before flat regions.

### Finding the Best Match (`findBestMatch`)
For each patch to fill, the algorithm searches for the most similar patch in the known region using a two-stage approach:

1. **Propagation** (`propagate`) — Checks if the offsets (displacement vectors) used by neighboring patches also yield a good match for the current patch. This exploits spatial coherence: nearby patches tend to have similar best matches.

2. **Random Hierarchical Search** — Explores the image at decreasing scales, starting with a broad search and progressively refining around the best candidate found so far.

Patch similarity is measured using **Sum of Squared Differences (SSD)**, computed only over known (unmasked) pixels. An early-exit optimization stops the SSD computation as soon as it exceeds the current best score.

### Patch Filling (`completePatch`)
Once the best matching source patch is found, its pixel values are copied into the target location. The mask is then updated to mark those pixels as known, and the priority queue is updated for all newly affected boundary patches.

### Pre-processing (`preprocessing`)
If the input image exceeds a maximum area of **360,000 pixels** (e.g., larger than ~600×600), it is downscaled proportionally to keep processing times reasonable.

### Post-processing (`postprocessing`)
A **3×3 Gaussian filter** is applied to the filled region to reduce block artifacts and create a smoother visual transition between the reconstructed area and its surroundings.

---

### Key Functions

| Function | Description |
|---|---|
| `MyCallBackFunc` | Mouse callback for interactive region selection |
| `computeMask` | Generates the binary inpainting mask |
| `isValidPatch` | Checks if a patch is fully in the known region |
| `isBoundaryPatch` | Detects patches at the mask boundary |
| `computePriority` | Computes fill priority for a boundary patch |
| `computeSSE` | SSD similarity metric with early-exit optimization |
| `propagate` | PatchMatch propagation step |
| `generateRandomPairs` | Random coordinate sampling for search |
| `findBestMatch` | Full nearest-neighbor search (propagate + random) |
| `completePatch` | Copies source patch pixels into the target region |
| `imageReconstruction` | Main inpainting loop with priority queue |
| `preprocessing` | Downscales oversized images |
| `postprocessing` | Applies Gaussian smoothing to the filled region |

---

## Requirements

- **C++17** or later
- **OpenCV 4.x** (core, imgproc, highgui modules)

### Build

1. Clone the repository.
2. Open the solution in Visual Studio.
3. Make sure OpenCV is linked correctly (include dirs + lib dirs + `.lib` files).
4. Build and run.

---

## Usage

1. **Run the application** — A menu will appear in the console.
2. **Load an image** — Select your `.bmp` file via the file dialog.
3. **Draw the region to remove** — Click and drag on the displayed image to select the area you want to fill.
4. **Wait for reconstruction** — The algorithm processes the region automatically. Progress is printed to the console.
5. **View the result** — The inpainted image is displayed. You can save it using the save option.

**Input format:** `.bmp`, color (RGB), any resolution (large images are auto-resized).  
**Output format:** `.bmp`, same dimensions and format as input.

---

## Results

Each test shows: **original image** → **selected region** → **reconstructed result**.

### Test 1 — Stacked Stones
Removing the topmost stone from a balanced rock arrangement. The algorithm fills it with sky and blurred background texture.

![Test 1 - Before](tests/test1_before.jpg)
![Test 1 - Selected Region](tests/test1_selected_region.jpg)
![Test 1 - After](tests/test1_after.jpg)

---

### Test 2 — Wooden Fence
Removing a fence post from a structured scene with strong repeating patterns.

![Test 2 - Before](tests/test2_before.jpg)
![Test 2 - Selected Region](tests/test2_selected_region.jpg)
![Test 2 - After](tests/test2_after.jpg)

---

### Test 3 — Lone Tree on a Hill
Removing a prominent tree from a landscape. The algorithm reconstructs the rolling green hills seamlessly.

![Test 3 - Before](tests/test3_before.jpg)
![Test 3 - Selected Region](tests/test3_selected_region.jpg)
![Test 3 - After](tests/test3_after.jpg)

---