import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

class ModeloAsistente:
    def __init__(self, ruta_dataset="datos/dataset.csv", umbral_confianza=0.55):
        self.ruta_dataset = ruta_dataset
        self.umbral_confianza = umbral_confianza
        self.vectorizador = CountVectorizer()
        self.modelo = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        self.entrenar()

    def entrenar(self):
        df = pd.read_csv(self.ruta_dataset)
        col_texto = df.columns[0]
        col_intencion = df.columns[1]

        X = self.vectorizador.fit_transform(df[col_texto].astype(str))
        y = df[col_intencion].astype(str)
        self.modelo.fit(X, y)

    def predecir(self, pregunta):
        X_nuevo = self.vectorizador.transform([pregunta])
        probabilidades = self.modelo.predict_proba(X_nuevo)[0]
        max_prob = max(probabilidades)
        indice = probabilidades.argmax()
        intencion = self.modelo.classes_[indice]

        if max_prob < self.umbral_confianza:
            return "desconocido", float(max_prob * 100)
        return intencion, float(max_prob * 100)
