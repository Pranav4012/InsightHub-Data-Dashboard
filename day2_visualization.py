import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/students.csv")

# ---- Plot 1: Attendance vs FinalScore ----
plt.figure()
plt.scatter(df["Attendance"], df["FinalScore"])
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.title("Attendance vs Final Score")
plt.show()

# ---- Plot 2: PracticeTime vs FinalScore ----
plt.figure()
plt.scatter(df["PracticeTime"], df["FinalScore"])
plt.xlabel("Practice Time (hours)")
plt.ylabel("Final Score")
plt.title("Practice Time vs Final Score")
plt.show()

# ---- Plot 3: StudyHours vs FinalScore ----
plt.figure()
plt.scatter(df["StudyHours"], df["FinalScore"])
plt.xlabel("Study Hours")
plt.ylabel("Final Score")
plt.title("Study Hours vs Final Score")
plt.show()

