## Deployment

The application is deployed using **Streamlit Community Cloud**, providing an interactive web interface for real-time poultry dropping classification.

### Running the Application Locally

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/poultry-flock-health-triage.git
cd poultry-flock-health-triage
```

2. Create and activate a virtual environment:

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Launch the Streamlit application:

```bash
streamlit run app/app.py
```

The application will open automatically in your default web browser.

---

### Using the Application

1. Upload a poultry dropping image in **JPG**, **JPEG**, or **PNG** format.
2. The application automatically preprocesses the image using the same validation pipeline employed during model evaluation.
3. The trained CNN with data augmentation performs inference.
4. The application displays:

   * Predicted class (Healthy or Unhealthy)
   * Prediction confidence
   * Class probability distribution

---

### Live Demo

The deployed application is available on Streamlit Community Cloud:

**🔗 Live Demo:** *(https://poultry-flock-health-triage-and-monitoring-system-dd75nfzend8p.streamlit.app/)*

---

### Deployment Platform

* **Frontend:** Streamlit
* **Inference Engine:** PyTorch
* **Deployment:** Streamlit Community Cloud
* **Model:** CNN with Data Augmentation
