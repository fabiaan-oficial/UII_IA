import sqlite3
from flask import Flask, render_template, request, jsonify
import base_datos as bd
from modelo import ModeloAsistente

app = Flask(__name__)

# Inicializar BD y entrenar modelo al arrancar
bd.crear_tablas()
bd.cargar_base_conocimiento("datos/respuestas.json")
ia = ModeloAsistente("datos/dataset.csv")

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
        return jsonify({"response": "Por favor escribe una consulta.", "respuesta": "Por favor escribe una consulta."})

    intencion, confianza = ia.predecir(pregunta)
    bd.registrar_consulta(pregunta, intencion, confianza)

    if intencion == "desconocido":
        bd.registrar_desconocida(pregunta)
        respuesta_texto = "No comprendí bien tu consulta o está fuera de los temas universitarios. Por favor acude a Atención a Alumnos."
        fuente = "Sistema de Respaldo (Fallback)"
    else:
        resultado = bd.obtener_respuesta_bd(intencion)
        if resultado:
            respuesta_texto, fuente = resultado
            respuesta_texto = f"{respuesta_texto} [Fuente: {fuente}]"
        else:
            respuesta_texto = "Intención detectada, pero aún no tiene respuesta en la base de datos."
            fuente = "Base de datos SQLite"

    return jsonify({
        "response": respuesta_texto,
        "respuesta": respuesta_texto,
        "intent": intencion,
        "intencion": intencion,
        "confidence": round(confianza, 2),
        "confianza": round(confianza, 2),
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
    print("  UII AI ASSISTANT + SQLITE CORRIENDO EN VIVO")
    print("  Abre en tu navegador: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(debug=False, port=5000)
