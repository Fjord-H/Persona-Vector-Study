# Persona Vectors Research Dashboard

Interactive dashboards for visualizing self-monitoring research across the complete research journey.

## Two Versions Available

### Dashboard v0 (Discovery Process)
**Port 8501** | Initial breakthrough and overfitting lessons

Shows the original discovery of tone vs content vectors and the reality check that exposed overfitting.

**Highlights:**
- Tone vectors: 38.5% accuracy
- Content vectors: 92.5% accuracy (small test set)
- The wake-up call: 44-61% on large test set

### Dashboard v1 (Complete Research)
**Port 8502** | Validated results across 3 models

Shows systematic validation with rigorous evaluation methodology.

**Highlights:**
- 3 models tested (GPT-2, Qwen, Llama)
- 200 training + 1,800 test examples
- Final results: 64-74% accuracy
- Comprehensive layer analysis + failed experiments

---

## Features

### Dashboard v0 Sections:
1. **Overview:** Initial discovery metrics
2. **Model Comparison:** GPT-2 vs Qwen (small scale)
3. **Results:** Original findings
4. **Layer Analysis:** Early layer testing
5. **Demo:** Interactive query testing

### Dashboard v1 Sections:
1. **Overview:** Complete research summary (3 models)
2. **The Discovery:** Tone vs content breakthrough story
3. **Reality Check:** Overfitting exposed
4. **Model Comparison:** Cross-model validation
5. **Layer Analysis:** Comprehensive layer testing
6. **Failed Experiments:** What didn't work (weighting, ensemble, categories)
7. **Demo:** Interactive classification with 18+ examples

---

## Installation
```bash
# Install requirements
pip install -r dashboard_requirements.txt
```

## Local Usage
```bash
# Run dashboard v0 (discovery)
streamlit run dashboard_v0.py --server.port 8501

# Run dashboard v1 (complete research)
streamlit run dashboard_v1.py --server.port 8502
```

Dashboards will open at:
- v0: `http://localhost:8501`
- v1: `http://localhost:8502`

---

## Docker Deployment

See `DOCKER.MD` for detailed Docker instructions.

**Quick start:**
```bash
# Run v0 (discovery)
docker run -d -p 8501:8501 --name dashboard-v0 fjordhauler/persona-vectors-dashboard:v0

# Run v1 (complete research)
docker run -d -p 8502:8502 --name dashboard-v1 fjordhauler/persona-vectors-dashboard:v1
```

---

## Live Demo

**Public dashboards available:**
- Dashboard v0 (Discovery): http://3.106.128.216:8501
- Dashboard v1 (Complete): http://3.106.128.216:8502

---

## For Interviews & Presentations

**Show both versions to demonstrate:**
- Scientific rigor (catching your own overfitting)
- Complete research journey (from discovery to validation)
- Systematic methodology (failed experiments documented)
- Production-ready insights (simple approaches win)

**Recommended flow:**
1. Start with v1 (shows final validated results)
2. Reference v0 (shows how you got there, including mistakes)
3. Emphasizes transparency and scientific honesty

---

## Technology Stack

- **Framework:** Streamlit
- **Visualization:** Matplotlib, Seaborn
- **Data:** Pandas, NumPy
- **Deployment:** Docker + EC2
- **Models:** GPT-2, Qwen, Llama (via Transformers)

---

## Notes

- Dashboards use pre-computed results from research notebooks
- Interactive demo uses simulated scores based on actual model behavior
- Full model inference requires loading weights (see notebooks)
- Perfect for presentations and portfolio demonstrations