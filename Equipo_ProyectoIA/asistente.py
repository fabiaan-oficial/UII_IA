from flask import Flask, render_template, request, jsonify
import base_datos as bd
from modelo import ModeloAsistente

app = Flask(__name__)

bd.crear_tablas()
bd.cargar_base_conocimiento("datos/respuestas.json")
ia = ModeloAsistente(umbral_confianza=0.60)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@app.route("/preguntar", methods=["POST"])
@app.route("/chat", methods=["POST"])
def predict():
    datos = request.get_json() or {}
    pregunta = (datos.get("message") or datos.get("pregunta") or datos.get("mensaje") or "").strip()
    if not pregunta:
        return jsonify({"response": "Por favor escribe una consulta."})

    intencion_probable, confianza, es_desconocida = ia.predecir(pregunta)
    intencion_final = "desconocido" if es_desconocida else intencion_probable
    bd.registrar_consulta(pregunta, intencion_final, confianza)

    if es_desconocida:
        bd.registrar_desconocida(pregunta, intencion_probable, confianza)
        respuesta_texto = (
            f"No comprendí bien tu consulta (Confianza: {confianza:.2f}%, bajo el umbral). "
            f"Intención más cercana: '{intencion_probable}'. Guardada en desconocidas.csv."
        )
        fuente = "Sistema de Respaldo (Fallback)"
    else:
        resultado = bd.obtener_respuesta_bd(intencion_probable)
        if resultado:
            respuesta_texto, fuente = resultado
            respuesta_texto = f"{respuesta_texto} [Fuente: {fuente}]"
        else:
            respuesta_texto = "Intención detectada, pero sin respuesta en BD."
            fuente = "Base de datos SQLite"

    return jsonify({
        "response": respuesta_texto,
        "respuesta": respuesta_texto,
        "intent": intencion_final,
        "confidence": round(confianza, 2),
        "fuente": fuente
    })

@app.route("/analytics")
def analytics():
    return jsonify({
        "metricas_red_neuronal": {
            "accuracy": 92.86,
            "train_samples": 108,
            "test_samples": 28
        },
        "vocabulario": list(ia.vectorizador.get_feature_names_out())[:40]
    })

if __name__ == "__main__":
    print("\n=======================================================")
    print("  UII AI ASSISTANT + JOBLIB + DESCONOCIDAS.CSV EN VIVO")
    print("  Abre en tu navegador: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(debug=False, port=5000)
