# 🏥 AI Disease Prediction System

> An AI-powered web application that predicts diseases based on user-reported symptoms — built for the *AI in Healthcare & Life Sciences Hackathon*.

---

## 📌 Problem Statement
Early disease detection is difficult and time-consuming. Patients often delay seeking help due to uncertainty about their symptoms. This system bridges that gap by using Machine Learning to provide instant, accessible preliminary predictions.

## 💡 Proposed Solution
A **Symptom → Disease Prediction** pipeline:
1. User selects symptoms via a clean web UI
2. A trained Random Forest model predicts the most likely disease
3. The app shows the prediction, confidence score, and suggested actions

---

## 🗂️ Project Structure

```
disease_prediction/
├── app.py               # Streamlit web application (main UI)
├── train_model.py       # ML model training script
├── dataset.csv          # Symptom-disease dataset (15 diseases, 18 symptoms)
├── model.pkl            # Trained Random Forest model (auto-generated)
├── label_encoder.pkl    # Label encoder for disease names (auto-generated)
├── symptom_cols.pkl     # Symptom column order (auto-generated)
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## ⚙️ Setup & Run

### 1. Clone / Download the project
```bash
cd disease_prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```
This generates `model.pkl`, `label_encoder.pkl`, and `symptom_cols.pkl`.

### 4. Launch the app
```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## 🧠 Technology Stack

| Layer        | Technology                  |
|--------------|-----------------------------|
| Language     | Python 3.10+                |
| ML Model     | Random Forest (Scikit-learn)|
| Data         | Pandas, NumPy               |
| UI           | Streamlit                   |
| Persistence  | Pickle                      |

---

## 🦠 Diseases Covered (15)

| #  | Disease          | #  | Disease        |
|----|------------------|----|----------------|
| 1  | Flu              | 9  | Migraine       |
| 2  | Common Cold      | 10 | Asthma         |
| 3  | COVID-19         | 11 | Heart Disease  |
| 4  | Pneumonia        | 12 | Anemia         |
| 5  | Malaria          | 13 | Chickenpox     |
| 6  | Dengue           | 14 | Measles        |
| 7  | Typhoid          | 15 | Tuberculosis   |
| 8  | Gastroenteritis  |    |                |

---

## 🔬 Symptoms Used (18)

Fever · Cough · Headache · Fatigue · Nausea · Chest Pain · Shortness of Breath · Sore Throat · Runny Nose · Body Ache · Vomiting · Diarrhea · Rash · Joint Pain · Dizziness · Loss of Appetite · Sweating · Chills

---

## 🚀 Future Scope

- 🩻 **Image-based diagnosis** — X-ray / MRI analysis using CNN (TensorFlow)
- 📱 **Mobile app** — React Native or Flutter frontend
- 🔴 **Real-time monitoring** — IoT wearable data integration
- 🌐 **Multilingual UI** — support for regional Indian languages
- 🏥 **Doctor referral system** — connect users to nearby specialists

---

## ⚠️ Disclaimer

This tool is for **educational and demonstration purposes only**. It is **not a substitute** for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

*Built by surojit using Python & Streamlit*
