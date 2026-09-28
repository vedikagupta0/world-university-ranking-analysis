# THE World University Rankings Analysis


Exploratory data analysis of the Times Higher Education World University Rankings dataset, 2k institutions across 115 countries.

A full write-up of the findings is in `ppt/world-university-rankings.pptx` (also as PDF). This repo holds the underlying data, cleaning pipeline, and notebook.



## What's here?

```
dataset/      world_university_rankings_2191.csv — raw dataset
notebooks/    world_university_rankings_eda_updated.ipynb — full analysis walkthrough
src/          eda_pipeline.py — reusable cleaning, stats, and plotting functions
images/       charts exported from the pipeline, used in the notebook and slides
ppt/          slide deck (PPTX + PDF) summarizing the key findings

```



## Setup

```bash
python -m venv venv
source venv/bin/activate      # venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```



## Usage

```bash
python src/eda_pipeline.py
```

Or open the notebook for the step-by-step analysis:

```bash
jupyter notebook notebooks/world_university_rankings_eda_updated.ipynb
```



## Key findings

* **Research is the dominant driver of overall score** (r = 0.90) — the strongest single predictor, ahead of Citations (0.86) and Teaching (0.81).
* **Internationalization** may prove to be a good aspect to build (r = 0.52 with share of international students) — a moderate correlation, weaker than the core academic pillars, but student diversity can be grown through admissions and recruitment far more quickly than research capacity.
* **The leading pack is tight**: Oxford (#1, 98.2) to Caltech (#7, 96.3) span less than 2 points.
* **Breadth ≠ quality nationally**: the US (171 ranked universities) and UK (109) rank 7th and 10th on average score, behind smaller systems like the Netherlands, Switzerland, and Belgium.
* **Student-staff ratio is not a useful quality signal** (r = −0.076), despite being a common admissions talking point.


See the slide deck or notebook conclusions for the full breakdown.



---
## Data source
Times Higher Education World University Rankings Official Website.

## License
See [LICENSE](LICENSE).

