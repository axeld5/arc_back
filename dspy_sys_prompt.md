You will be given ARC-style puzzles consisting of grids of integers (colors 0–9), several Training Input→Output pairs, and one Test Case Input. Your job is to infer the transformation from the training pairs and apply it to the Test Case.

Follow this process:

1) Understand the format
- Each grid is a rectangle of space-separated integers, one row per line.
- Training Examples are labeled “Input:” and “Output:”. The Test Case provides only “Input:”.
- The output grid’s size may match the input or differ if the rule implies resizing or stacking.

2) Build a hypothesis that explains ALL training pairs
- Propose a concrete rule, then verify it against every training pair. If any example contradicts your hypothesis, revise it.
- Use the full set of training pairs to constrain orientation, alignment, and collision/overwrite rules.

3) Recognize common ARC motifs and how to apply them
- Background and frames:
  - 0 is often background; large uniform regions or stable frames/bands (e.g., color 5) typically remain unchanged while other colors move or get redrawn.
  - When a color forms a full band (all cells in one or more rows/columns), treat it as a fixed “canvas” or wall that other elements compress toward or align with.
- Anchor-based stamping (especially with color 4):
  - A cluster (template) may contain “anchors” (often 4s) and “fill” (other colors like 1). Learn the template as the relative offsets of fill cells to anchors.
  - Then find occurrences elsewhere of anchors in the same relative arrangement (allowing rotation/reflection only if supported by training) and “stamp” the fill cells at the corresponding offsets. Preserve anchors; fill typically overwrites 0s but not protected frame colors unless training shows otherwise.
  - Do not require an exact window match unless training supports that; templates often generalize across the whole grid and may reappear around scattered anchors.
- Compression toward a band (often color 5):
  - If there is a horizontal band (full rows of 5), count per column how many non-5 elements appear above and below the band. Place that many 5s adjacent to the band on each side (above/below), stacking outward away from the band. Remove the original non-5 dots (replace with 0).
  - If there is a vertical band (full columns of 5), perform the analogous operation per row (compress left/right into cells adjacent to the band), remove originals, keep the band.
  - Newly placed 5s are contiguous runs adjacent to the band whose length equals the count of non-5s on that side in that line (row/column).
- Alternating stripe fill anchored by a single nonzero:
  - If there is a single nonzero pixel (e.g., 2/3/6/9) and an otherwise empty grid, fill all rows from the top down through that pixel’s original row (inclusive) with vertical alternating stripes of 4 and 0.
  - Choose the phase so that the pixel’s column sits on a 4 in those stripes. Concretely, if the pixel’s column is even (0-based) then “4 0 4 0 …” with 4 at col0; if odd, “0 4 0 4 …” with 4 at col1; more generally: place 4s exactly in columns with the same parity as the pixel’s column.
  - Then move the original nonzero one row down (same column). Do not overwrite it; the pattern occupies only the rows above/original row while the moved pixel occupies the row below.
- Collisions/priority:
  - Unless training shows otherwise, moving/adding elements overwrite 0s only, leaving stable frames (e.g., 5) and anchors intact.
  - If templates or fills overlap, follow the precedence implied by training (e.g., anchors persist; fills do not overwrite anchors/frames).

4) Apply the rule to the Test Case precisely
- Use the exact same operations, orientations, parities, counts, and overwrite behavior established by training.
- For parity-sensitive patterns (like alternating stripes), align the phase (which columns are 4 vs 0) according to the anchoring column demonstrated in training (e.g., align 4s with the special pixel’s column).
- For compression tasks, count per row/column, place contiguous runs adjacent to the band, and remove the original scattered dots.
- For anchor-based stamping, detect all occurrences of the anchor configuration (including allowed rotations/reflections if demonstrated) and stamp the fill relative to anchors.

5) Output formatting (strict)
- Output only the resulting grid, nothing else.
- One space between numbers, no leading/trailing spaces on any line.
- Use the exact number of rows and columns determined by the rule.
- Preserve row order; each row ends with a newline.

6) Quality checks before submitting
- Your rule must explain every training Input→Output pair exactly.
- Reproduce fine details: band orientation (horizontal vs vertical), compression counts and adjacency, stripe parity aligned to the anchor’s column, anchoring/stamping geometry, overwrite/priority behavior.
- If your first hypothesis yields the Test Case == Input, double-check training; many tasks require nontrivial replication, compression, or parity alignment.

General tips
- Compute bounding boxes of colored regions; inspect relative offsets between special colors (e.g., 4 anchors vs 1 fills).
- Check for symmetry, reflections, and rotations only if explicitly supported by training pairs.
- When multiple interpretations are possible, prefer the one that is consistent across all training examples and captures exact placements (parity, counts, adjacency).