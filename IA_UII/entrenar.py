import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

def entrenar():
    df = pd.read_csv('dataset.csv').dropna()
    col_texto = 'texto' if 'texto' in df.columns else 'pregunta'
    col_intencion = 'intencion' if 'intencion' in df.columns else 'respuesta'

    X = df[col_texto].astype(str)
    y = df[col_intencion].astype(str)

    # División 80% Entrenamiento y 20% Prueba conservando la proporción de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Pipeline: Bag of Words -> TF-IDF -> Red Neuronal MLP con solver 'lbfgs' (óptimo para N < 1000)
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(
            lowercase=True,
            ngram_range=(1, 2)
        )),
        ('tfidf', TfidfTransformer()),
        ('classifier', MLPClassifier(
            hidden_layer_sizes=(64,),
            activation='relu',
            solver='lbfgs',      # 'lbfgs' converge de manera óptima y precisa en datasets pequeños
            alpha=0.01,
            max_iter=1000,
            random_state=42
        ))
    ])

    # Entrenar la Red Neuronal
    pipeline.fit(X_train, y_train)

    # Evaluar desempeño en conjunto de prueba (X_test, y_test)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("\n=======================================================")
    print("      ENTRENAMIENTO RED NEURONAL (MLPClassifier)")
    print("=======================================================")
    print(f"-> Muestras X_train: {len(X_train)} | Muestras X_test: {len(X_test)}")
    print(f"-> Accuracy (Exactitud en Test): {accuracy * 100:.2f}%")
    print("\nReporte de Clasificación en prueba:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("=======================================================\n")

    # Guardar modelo y métricas
    joblib.dump(pipeline, 'modelo_uii_assistant.pkl')
    
    metrics = {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': round(accuracy * 100, 2)
    }
    joblib.dump(metrics, 'metricas_modelo.pkl')

    return pipeline

if __name__ == '__main__':
    entrenar()
