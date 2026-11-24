import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Cargar dataset
df = pd.read_csv("datos/dataset.csv")
X_texto = df[df.columns[0]].astype(str)
y = df[df.columns[1]].astype(str)

X_train_txt, X_test_txt, y_train, y_test = train_test_split(
    X_texto, y, test_size=0.20, random_state=42, stratify=y
)

configuraciones = [
    ("Exp 1: CountVectorizer + MLP (32,)", CountVectorizer(), (32,)),
    ("Exp 2: CountVectorizer + MLP (64, 32)", CountVectorizer(), (64, 32)),
    ("Exp 3: TfidfVectorizer + MLP (32,)", TfidfVectorizer(), (32,)),
    ("Exp 4: TfidfVectorizer + MLP (64, 32)", TfidfVectorizer(), (64, 32)),
]

resultados = []
mejor_f1 = -1
mejor_modelo = None
mejor_vec = None
mejor_nombre = ""

print("\n==========================================================================================")
print("   COMPARACIÓN DE CONFIGURACIONES: VECTORIZADORES Y ARQUITECTURAS MLPClassifier")
print("==========================================================================================")
print(f"{'Experimento':<40} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 90)

for nombre, vectorizador, capas in configuraciones:
    X_train = vectorizador.fit_transform(X_train_txt)
    X_test = vectorizador.transform(X_test_txt)

    modelo = MLPClassifier(hidden_layer_sizes=capas, max_iter=1500, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    prec, rec, f1 = prec * 100, rec * 100, f1 * 100

    print(f"{nombre:<40} | {acc:>8.2f}% | {prec:>8.2f}% | {rec:>8.2f}% | {f1:>8.2f}%")

    if f1 >= mejor_f1:
        mejor_f1 = f1
        mejor_modelo = modelo
        mejor_vec = vectorizador
        mejor_nombre = nombre

print("=" * 90)
print(f"\n-> Configuración ganadora: {mejor_nombre} (F1-Score: {mejor_f1:.2f}%)")

# Entrenar la configuración ganadora con el 100% del dataset antes de guardar con joblib
X_full = mejor_vec.fit_transform(X_texto)
mejor_modelo.fit(X_full, y)

joblib.dump(mejor_modelo, "modelo_guardado.joblib")
joblib.dump(mejor_vec, "vectorizador_guardado.joblib")
print("-> Guardados con éxito: 'modelo_guardado.joblib' y 'vectorizador_guardado.joblib'\n")
