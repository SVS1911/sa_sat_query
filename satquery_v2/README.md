# SatQuery AI

An interactive vision-language assistant for multimodal remote-sensing image
analysis through text queries.

Upload satellite imagery, ask a question in plain English. SatQuery routes the
task, runs a fine-tuned Qwen3-VL model over the imagery, measures the pixels
itself, and returns an evidence-grounded answer with a visible execution trace.

| Workflow | Input | Example question |
|---|---|---|
| Single image | one scene | *What is present in this image?* |
| Change detection | T1 + T2 | *Calculate the percentage of changed area.* |
| Optical + SAR | optical + radar pair | *Compare these images and explain what they show.* |

---

## Read this first: your token

A Hugging Face token was shared in plain text during development. **Treat it as
compromised.** Revoke it at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), issue
a new one, and put the new value in `.env`, which is git-ignored.

A leaked read token can pull every private repo your account can see. Nothing in
this codebase ever prints a token, logs it, or returns it from the API — the
`/api/model` endpoint reports only `token: set` or `token: none`.

---

## The rule this project is built around

**A model may describe a change. It may not measure one.**

Semantic content — what the scene is, which land-cover classes are present —
comes from the fine-tuned model. Every *number* — pixel counts, changed area,
region counts, class breakdowns — is computed with NumPy from your actual image
data.

The two are never blended. Each answer ends with a provenance line:

```
Source: fine-tuned vision-language model + measured pixel statistics
```

If the model cannot load, the measurements survive intact, the answer says so,
and nothing is invented to fill the gap.

### Two kinds of confidence

When the model answers, SatQuery reports **generation confidence**: the mean
probability the decoder assigned to the tokens it actually produced. That is a
real measurement of how sure the model was of its own wording. It is *not* a
class probability, and the interface never relabels it as one.

When the local specialists answer instead, the confidence is a heuristic based
on how dominant the leading land-cover class is. The results panel tells you
which one you are looking at.

---

## Why the model runs locally

`aanandmodi/satquery-qwen3vl-bigearthnet-txt-lora` is a PEFT LoRA adapter on a
Qwen3-VL base. Hugging Face's serverless Inference API does not serve arbitrary
LoRA adapters on multimodal bases, so there is no free hosted endpoint to call.
The realistic options are a paid dedicated Inference Endpoint, or loading the
weights in-process. This app does the latter: it works on a laptop with a GPU,
in Colab, and on a lab workstation with no extra billing.

Consequences the code handles explicitly:

* **Weights load lazily, in a background thread.** The server starts in under a
  second and the UI shows real load progress instead of hanging.
* **The base model is discovered, not guessed.** The adapter records what it was
  trained on in `adapter_config.json`; SatQuery reads that. Set
  `HF_BASE_MODEL_ID` only if you want to override it.
* **A GPU is not assumed.** CPU inference works but is slow enough that the
  status panel warns you rather than letting you wonder.

### Hardware

| Setup | VRAM | Notes |
|---|---|---|
| CUDA GPU, `HF_LOAD_IN_4BIT=1` | ~6 GB | Recommended for an 8B base |
| CUDA GPU, bf16 | ~16–18 GB | Fastest |
| Apple Silicon (`mps`) | unified | Works; slower than CUDA |
| CPU only | — | Minutes per answer. Fine for a smoke test |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

**Install PyTorch first.** It is deliberately absent from `requirements.txt`,
because the right wheel depends on your CUDA version and pip would otherwise
put a CPU-only build on a GPU machine.

```bash
# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
# or CPU only
pip install torch
```

Then everything else:

```bash
pip install -r requirements.txt
cp .env.example .env               # then add your new HF_TOKEN
```

### Check the model before starting the server

```bash
python tools/test_model.py
```

Four stages: dependencies, weight loading, image preparation, inference. It
prints what the model actually returns for a satellite image, so a failure here
isolates the problem to the model rather than the app.

```bash
python tools/test_model.py --4bit --verbose
python tools/test_model.py --image data/sample/optical_t1.png
```

### Run

```bash
python app.py
```

Open <http://127.0.0.1:8000>. During development, `uvicorn app:app --reload`
gives you auto-reload.

To try the interface before the weights are ready, run with the model off:

```bash
HF_ENABLED=0 python app.py
```

Everything works; answers are labelled `Source: local specialist`.

---

## Interface

A single-page console with client-side routing, so navigating away from the
console does not discard your uploaded imagery or your last result.

* **Home** (`#/`) — landing page. An orbital-pass animation plays once on load.
* **Console** (`#/console`) — pick a workflow, drop imagery, ask a question.
* **Model** (`#/model`) — live state, load progress, resolved base model, device.
* **Guide** (`#/guide`) — how the pipeline works.
* **API** (`/api/docs`) — interactive OpenAPI docs.

Transitions use [anime.js](https://animejs.com) v4, vendored at
`web/static/js/anime.esm.min.js` so the app has no CDN dependency at runtime.
Motion is one directional wipe per navigation plus a short stagger on the
incoming panels; `prefers-reduced-motion` collapses all of it to instant swaps.

Colour encodes function rather than decoration: cyan for optical, amber for
radar, and magenta reserved exclusively for detected change — so a magenta pixel
anywhere in the interface means one thing.

---

## API

Everything the console does is available as JSON.

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/api/health` | Liveness. Answers while the model is still loading |
| `GET`  | `/api/model` | Model state, progress, config (never the token) |
| `POST` | `/api/model/load` | Start or retry loading |
| `GET`  | `/api/options` | Workflows, band presets, example queries |
| `POST` | `/api/analyze` | Run an analysis |

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F mode=single \
  -F "query=What is present in this image?" \
  -F image_a=@scene.tif

curl -X POST http://127.0.0.1:8000/api/analyze \
  -F mode=change \
  -F "query=Calculate the percentage of changed area." \
  -F image_a=@t1.tif -F image_b=@t2.tif
```

The response carries `answer`, `measurements`, `trace`, `evidence` (a PNG data
URI), `confidence` with its `confidence_kind`, and `model_source`. Rejections
use the same envelope with `success: false`, so a client needs one rendering
path rather than two.

---

## Architecture

```
                    USER
          image(s) + natural-language question
                     |
                     v
     web/  ──────────┴──────────  single-page console, anime.js transitions
                     |  POST /api/analyze
                     v
            controller/input_validator.py     format, bands, footprint, CRS
                     |
                     v
            controller/query_parser.py        router / planner -> task
                     |
        +------------+------------+
        v            v            v
     single      bi-temporal   optical+SAR     controller/agentic_controller.py
        |            |            |            (executor)
        +------------+------------+
                     v
         api/hf_model.py                       Qwen3-VL + LoRA (semantics)
                     +
         utils/numerical.py                    NumPy measurements (quantities)
                     v
         controller/synthesizer.py             answer + provenance
                     v
         utils/evidence.py                     labelled evidence board
                     v
         controller/execution_trace.py         observable step trace
```

The trace records **what the system did** — which validator ran, which workflow
was chosen, how long each stage took. It is not a reasoning log, and no internal
chain-of-thought is exposed.

### Visual evidence

| Workflow | Panels |
|---|---|
| Single image | Original · Land-cover classification |
| Change detection | T1 · T2 · Difference magnitude · Change mask |
| Optical + SAR | Optical · SAR · Radar structure · Fused result |

The change mask uses magenta on a desaturated base rather than the conventional
red/green, so it stays legible under deuteranopia.

---

## Band layout matters more than anything else

Land-cover classes are defined by spectral indices (NDVI, NDWI, MNDWI, NDBI),
not by colour. Without a NIR or SWIR band, shadow reads as water and bright soil
reads as buildings. On the built-in benchmark, supplying the correct band roles
moves overall accuracy from 64.8% to 93.8% and mean IoU from 45.2% to 85.6%.

Set **Band layout of your source** in the console to match how your bands are
stacked. Leave it unset and SatQuery falls back to RGB colour proxies and says
so in the trace, rather than guessing silently.

```bash
python tests/test_accuracy_fix.py     # reproduces those figures
```

---

## Tests

```bash
HF_ENABLED=0 python -m unittest discover -s tests -p "test*.py"
HF_ENABLED=0 PYTHONPATH=. python tests/api_e2e.py     # all three workflows + error paths
```

Both run offline and need no model weights.

---

## Known limitations

* **First run downloads several GB.** Later runs read the local Hugging Face
  cache. Set `HF_LOCAL_ONLY=1` to forbid network access once cached.
* **Fusion is computed locally.** The model analyses one image at a time, so for
  optical+SAR it is called once per modality and the *fused* result comes from
  `models/optical_sar_fusion.py`. The answer states this explicitly.
* **`extract_labels` only recognises vocabulary the model produced.** It never
  adds a BigEarthNet class the model did not mention, so the label list can
  never be richer than the model's actual output.
* **Co-registration is checked by size and CRS, not by content.** Mismatched
  sizes are resampled; mismatched projections are refused. True geometric
  co-registration is assumed to have happened upstream.
* **Region counts are approximate on very large masks.** Above ~262k pixels the
  mask is downsampled before labelling, and the count is marked approximate.
* **Physical area needs georeferencing.** Without a CRS and resolution, SatQuery
  reports pixels and says why, rather than inventing hectares.

---

## Repository layout

```
app.py                          FastAPI backend + static host
api/
  hf_config.py                  environment-driven configuration
  hf_model.py                   Qwen3-VL + LoRA runner (lazy, thread-safe)
controller/
  agentic_controller.py         orchestrator (validator -> router -> executor)
  input_validator.py            format / footprint / CRS checks
  query_parser.py               natural language -> task routing
  synthesizer.py                answer assembly + provenance
  execution_trace.py            observable step trace
models/                         local specialists (VQA, caption, grounding,
                                change analysis, optical/SAR fusion)
utils/
  image_io.py                   loading, normalisation, alignment, GSD
  numerical.py                  all measurements
  evidence.py                   labelled evidence boards
  spectral_indices.py           NDVI / NDWI / MNDWI / NDBI classification
  visualization.py              overlays and masks
  evaluation.py                 accuracy harness
web/
  index.html                    single-page console
  static/css/app.css            mission-control theme
  static/js/router.js           hash router + anime.js transitions
  static/js/main.js             console logic, model polling
  static/js/anime.esm.min.js    vendored anime.js v4.5.0 (MIT)
tools/test_model.py             standalone model check
```

`.env`, generated reports, uploads, and model caches stay out of version
control.
