# Student Performance Prediction System

Machine Learning capstone project that analyzes student academic data, predicts exam performance, and generates personalized improvement recommendations.

**Author:-** Soham Mukherjee, B.Tech CSE (Data Science), Heritage Institute of Technology, Kolkata

**Internship and Training:-** SkillOrbit , Machine Learning

**Live link:-** https://student-performance-prediction-system132.streamlit.app/

**Demo Video:-** https://drive.google.com/file/d/1zDMxPowsAiJavn6AWWzaEHUs2teyXcTc/view?usp=drive_link

**Demo Video**- https://drive.google.com/file/d/1zDMxPowsAiJavn6AWWzaEHUs2teyXcTc/view?usp=drive_link
## What's in this repo

| File | Description |
|---|---|
| `Student_Performance_Prediction_System.ipynb` | Main notebook - data preprocessing, EDA, model training/evaluation, dashboard preview, recommendation engine |
| `dashboard.py` | Interactive Streamlit dashboard |
| `StudentPerformanceFactors.csv` | Dataset (6,607 student records, 20 columns) |
| `Project_Report.docx` | Full written project report |
| `Student_Performance_Prediction_System.pptx` | Presentation deck |
| `requirements.txt` | Python dependencies |

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook Student_Performance_Prediction_System.ipynb
```

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Make sure `StudentPerformanceFactors.csv` is in the same folder as `dashboard.py`.

## Results summary

**Regression (predicting exact Exam_Score):**

| Model | MAE | RMSE | R2 Score |
|---|---|---|---|
| Linear Regression | 0.444 | 1.799 | 0.771 |
| Random Forest | 1.228 | 2.243 | 0.644 |
| Decision Tree | 1.645 | 2.676 | 0.493 |

**Classification (predicting Low / Average / High performance level):**

| Model | Accuracy |
|---|---|
| Random Forest | 78.97% |
| Decision Tree | 71.94% |

Attendance, Hours_Studied, and Previous_Scores are consistently the strongest predictors across both tasks.

## Scope

Per the project brief, this system intentionally does not use deep learning, real-time monitoring, or large-scale platform architecture - it stays focused on a clean, functional prediction pipeline (Linear Regression / Decision Tree / Random Forest).
