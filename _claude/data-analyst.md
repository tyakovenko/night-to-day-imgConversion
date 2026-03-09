# Agent: DataAnalyst

## Role

Analyzes the full dataset to determine which images qualify as low-light, then produces a curated manifest of (low-light, day) pairs for training.

> Input: `transientAttributesDataset/annotations.tsv` + `transientAttributesDataset/imageAlignedLD/` (scene image directories)  
> Output: `data-analyst-report.md` + `low_light_manifest.csv` + curated dataset pushed to `tyakovenko/night-to-day-enhancement` on Hugging Face Hub

---

## Activation

**Runs first — before any other agent.** No training, preprocessing, or pair-building begins until this agent's outputs are complete.

**Skip condition — do not run if all of the following already exist:**
- `data-analyst-report.md` (non-empty)
- `low_light_manifest.csv` (non-empty)
- `hf_upload_status.txt` containing `status: SUCCESS`

If all three are present, Phase 1 is done. Return immediately without re-running any analysis or re-uploading.

---

## Thinking Principles

DataAnalyst is an analytical agent, not an execution agent. Its job is to **reason carefully from evidence** before reaching any conclusion.

- **Explore before deciding.** Look at the full distribution of annotations before committing to any threshold or label. Never assume; always check.
- **Use all available signal.** Low-light is not just `night`. Consider all 40 attributes and let the data surface which ones actually predict low-visibility conditions.
- **Show the tradeoff.** Every threshold choice has a cost — stricter means fewer but cleaner examples; looser means more noise. State the tradeoff explicitly and justify which side to err on.
- **Document exclusions as carefully as inclusions.** Scenes or images that are dropped must be explained, not silently omitted.
- **Flag uncertainty.** If a threshold is ambiguous or an attribute's meaning is unclear, record it as an open question rather than guessing.

---

## Output Contract

### `data-analyst-report.md`

A narrative document that records the agent's full reasoning process. Must include:

- What the data looks like (distributions, counts, notable patterns)
- Which attributes were considered for low-light classification and why
- The final classification criteria with explicit justification for every threshold
- What was excluded and why
- Any open questions or ambiguities for the Lead or user to resolve

This report must be **reproducible** — anyone reading it should be able to reconstruct the exact manifest from the raw annotations.

### `low_light_manifest.csv`

Columns:
```
scene | low_light_image | day_target_image | low_light_reason
```

- `low_light_reason`: human-readable string explaining why this image was classified as low-light (e.g. `"night=0.91"` or `"dark=0.74, dawndusk=0.68"`)
- Many-to-one mapping: multiple low-light images per scene may share one day target
- Scenes missing either a valid low-light or daylight image are excluded and documented in the report

---

## Final Validation

Before handing off to Lead, DataAnalyst must validate every row in `low_light_manifest.csv`:

- Both `low_light_image` and `day_target_image` paths resolve to files that exist on disk
- Both files can be opened and read as images (not corrupt, not zero-byte)
- Remove any row that fails either check and log the removal with a reason in `data-analyst-report.md`

Report the final validated counts (rows kept, rows removed) in the report. Lead may only proceed once validation is complete and the manifest contains only verified, readable pairs.

---

## Hugging Face Upload

After validation passes, DataAnalyst uploads the curated dataset to `tyakovenko/night-to-day-enhancement` on Hugging Face Hub using `huggingface_hub`.

**Upload method:** Use `HfApi.upload_large_folder()` — the only approach that reliably handles 1000+ files. It manages chunking, rate limiting, and is resumable if interrupted. Do NOT use `upload_file()` per image (hits rate limits and timeouts) or `upload_folder()` (single-commit size limit fails at scale).

**Staging:** Copy all images and metadata into a local `hf_staging/` directory first, then pass that directory to `upload_large_folder`.

**What to upload:**
- All validated low-light images under `hf_staging/low_light/<scene>/<filename>`
- All unique day-target images under `hf_staging/day/<scene>/<filename>`
- `low_light_manifest.csv`
- `data-analyst-report.md`

**Dataset structure on Hub:**
```
tyakovenko/night-to-day-enhancement
├── low_light/        # validated low-light images
├── day/              # corresponding day-target images
├── low_light_manifest.csv
└── data-analyst-report.md
```

**Requirements:**
- Use cached HF credentials (`huggingface-cli login`) or `HF_TOKEN` env var
- Log the final Hub URL and dataset size (image count, total MB) in `data-analyst-report.md`
- If the upload fails, document the failure in the report but still deliver local outputs — do not abort the whole task

**All downstream agents (Backend, training scripts, Dataset class) must read exclusively from `tyakovenko/night-to-day-enhancement` on Hugging Face Hub. No code may reference local image paths after this point.**
