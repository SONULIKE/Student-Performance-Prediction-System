import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# ---------------------------------------------------------------
# Data loading + the same preprocessing used in the notebook
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("StudentPerformanceFactors.csv")
    df.drop_duplicates(inplace=True)

    for col in ["Teacher_Quality", "Parental_Education_Level", "Distance_from_Home"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df_display = df.copy()

    ordinal_map = {"Low": 0, "Medium": 1, "High": 2}
    for col in ["Parental_Involvement", "Access_to_Resources", "Motivation_Level",
                "Family_Income", "Teacher_Quality"]:
        df[col] = df[col].map(ordinal_map)

    df["Parental_Education_Level"] = df["Parental_Education_Level"].map(
        {"High School": 0, "College": 1, "Postgraduate": 2})
    df["Distance_from_Home"] = df["Distance_from_Home"].map(
        {"Near": 0, "Moderate": 1, "Far": 2})
    df["Peer_Influence"] = df["Peer_Influence"].map(
        {"Negative": 0, "Neutral": 1, "Positive": 2})

    binary_map = {"Yes": 1, "No": 0}
    for col in ["Extracurricular_Activities", "Internet_Access", "Learning_Disabilities"]:
        df[col] = df[col].map(binary_map)

    df["School_Type"] = df["School_Type"].map({"Public": 0, "Private": 1})
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    df["Performance_Level"] = pd.qcut(df["Exam_Score"], q=3, labels=["Low", "Average", "High"])

    return df, df_display


@st.cache_resource
def train_models(df):
    features = [c for c in df.columns if c not in ["Exam_Score", "Performance_Level"]]
    X = df[features]

    y_reg = df["Exam_Score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    y_clf = df["Performance_Level"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(Xc_train, yc_train)

    return reg, clf, features, (X_test, y_test)


def generate_recommendations(student, medians, prev_median):
    tips = []
    if student["Hours_Studied"] < medians["Hours_Studied"]:
        tips.append("Increase weekly study hours - currently below the class median.")
    if student["Attendance"] < medians["Attendance"]:
        tips.append("Improve class attendance to stay on top of coursework.")
    if student["Previous_Scores"] < prev_median:
        tips.append("Focus on revising foundational topics - previous scores are below average.")
    if student["Tutoring_Sessions"] < medians["Tutoring_Sessions"]:
        tips.append("Attend more tutoring/practice sessions for extra support.")
    if student["Sleep_Hours"] < 6:
        tips.append("Aim for more consistent sleep - it's currently on the lower side.")
    if student["Motivation_Level"] == "Low":
        tips.append("Set smaller, achievable goals to help build study motivation.")
    if not tips:
        tips.append("Keep up the current routine - performance indicators look healthy.")
    return tips


df, df_display = load_data()
reg_model, clf_model, features, (X_test, y_test) = train_models(df)

# ---------------------------------------------------------------
# Sidebar - student selector
# ---------------------------------------------------------------
st.sidebar.title("Student Performance Prediction System")
student_idx = st.sidebar.selectbox("Select a student (row index)", df_display.index.tolist())
student_row = df.loc[student_idx]
student_display = df_display.loc[student_idx]

st.title("📊 Student Performance Dashboard")

# ---------------------------------------------------------------
# Top row - predicted score + performance level for the selected student
# ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)

pred_score = reg_model.predict(student_row[features].values.reshape(1, -1))[0]
pred_level = clf_model.predict(student_row[features].values.reshape(1, -1))[0]

col1.metric("Actual Exam Score", f"{student_display['Exam_Score']}")
col2.metric("Predicted Exam Score", f"{pred_score:.1f}")
col3.metric("Predicted Performance Level", pred_level)

st.divider()

# ---------------------------------------------------------------
# Student comparison + factor-wise analysis
# ---------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Student vs Class Average")
    compare_cols = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores", "Tutoring_Sessions"]
    compare_df = pd.DataFrame({
        "This Student": student_display[compare_cols],
        "Class Average": df_display[compare_cols].mean()
    })
    st.bar_chart(compare_df)

with right:
    st.subheader("Performance Level Breakdown (All Students)")
    st.bar_chart(df["Performance_Level"].value_counts().reindex(["Low", "Average", "High"]))

st.subheader("Performance Trends")
trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_display, x="Attendance", y="Exam_Score", alpha=0.4, ax=ax)
    ax.axvline(student_display["Attendance"], color="red", linestyle="--", label="This student")
    ax.legend()
    ax.set_title("Attendance vs Exam Score")
    st.pyplot(fig)

with trend_col2:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df_display, x="Hours_Studied", y="Exam_Score", alpha=0.4, ax=ax2, color="darkorange")
    ax2.axvline(student_display["Hours_Studied"], color="red", linestyle="--", label="This student")
    ax2.legend()
    ax2.set_title("Study Hours vs Exam Score")
    st.pyplot(fig2)

st.divider()

# ---------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------
st.subheader("Personalized Recommendations")
medians = df_display[["Hours_Studied", "Attendance", "Sleep_Hours", "Tutoring_Sessions"]].median()
prev_median = df_display["Previous_Scores"].median()

for tip in generate_recommendations(student_display, medians, prev_median):
    st.write("- " + tip)

st.caption("Run with: streamlit run dashboard.py (requires StudentPerformanceFactors.csv in the same folder)")
