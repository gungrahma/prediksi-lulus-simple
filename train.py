# Model Decision Tree
from sklearn.tree import DecisionTreeClassifier
import joblib

# Format X -> [jam_belajar, ikut_bimbel (1 = ikut bimbel/iya, 0 = tidak ikut/tidak)]
X = [
    [2, 0], [3, 0], [5, 1], [7, 1],
    [1, 0], [4, 1], [6, 0], [8, 0]
]

# Format Y atau target jawaban -> [1 = lulus, 0 = tidak]
y = [0, 0, 1, 1, 0, 1, 1, 1]

model = DecisionTreeClassifier()

print("Mulai latihan model DecisionTree")
model.fit(X,y)

joblib.dump(model, "model_kelulusan.joblib")
print("Model train sukses! disimpan sebagai model_kelulusan.joblib")