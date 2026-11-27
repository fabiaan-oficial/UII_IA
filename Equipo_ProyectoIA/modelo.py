import os
import joblib

class ModeloAsistente:
    def __init__(self, umbral_confianza=0.60):
        self.umbral_confianza = umbral_confianza
        if not os.path.exists("modelo_guardado.joblib") or not os.path.exists("vectorizador_guardado.joblib"):
            raise FileNotFoundError("Ejecuta primero 'python3 experimentos.py' para generar los archivos .joblib")
        self.modelo = joblib.load("modelo_guardado.joblib")
        self.vectorizador = joblib.load("vectorizador_guardado.joblib")

    def predecir(self, pregunta):
        X_nuevo = self.vectorizador.transform([pregunta])
        probabilidades = self.modelo.predict_proba(X_nuevo)[0]
        max_prob = float(max(probabilidades))
        indice = probabilidades.argmax()
        intencion_probable = str(self.modelo.classes_[indice])

        # Se rechaza si el modelo la reconoce como ajena al instituto, si no está
        # seguro, o si el texto no tiene nada parecido a lo aprendido
        es_desconocida = (
            intencion_probable == "fuera_de_alcance"
            or max_prob < self.umbral_confianza
            or X_nuevo.nnz == 0
        )
        return intencion_probable, max_prob * 100, es_desconocida
