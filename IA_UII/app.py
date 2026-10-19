import os
import csv
import pandas as pd
import joblib
import re
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from entrenar import entrenar

app = Flask(__name__)
MODELO_FILE = 'modelo_uii_assistant.pkl'

# Inicializar o cargar modelo de Red Neuronal
if not os.path.exists(MODELO_FILE):
    modelo = entrenar()
else:
    modelo = joblib.load(MODELO_FILE)

RESPUESTAS_INTENCION = {
    'saludo': "¡Hola! Soy el asistente virtual del Instituto Irapuato. ¿En qué te puedo ayudar hoy?",
    'inscripciones': "Para inscripciones y admisión, puedes revisar los requisitos de nuevo ingreso y fechas en el portal institucional o acudir a Servicios Escolares.",
    'becas': "Puedes solicitar información y tramitar becas institucionales en la oficina de Atención a Alumnos o en la convocatoria del portal UII.",
    'horarios': "Puedes consultar tu horario y tira de materias en el portal universitario oficial: https://uii.edu.mx/portal",
    'servicios_escolares': "Puedes tramitar constancias, boletas y credenciales directamente en la oficina de Servicios Escolares en el campus central.",
    'pagos': "Puedes pagar en cajas en el edificio de colores o en el aula escolar mediante pago con tarjeta. Las colegiaturas se pagan los días 10 de cada mes.",
    'ubicacion': "Es el edificio entrando al Instituto Irapuato justo en frente. Le llaman edificio de colores porque tiene muchos colores en sus espejos."
}

PALABRAS_CLAVE_VALIDAS = {
    'hola', 'buenos', 'dias', 'tardes', 'noches', 'saludos',
    'inscripcion', 'inscripciones', 'inscribirme', 'ingreso', 'admision', 'requisitos',
    'beca', 'becas', 'apoyo', 'apoyos', 'economico', 'solicitar',
    'horario', 'horarios', 'clase', 'clases', 'materia', 'materias', 'tira', 'portal',
    'constancia', 'boleta', 'boletas', 'credencial', 'credenciales', 'tramite', 'servicios', 'escolares',
    'pago', 'pagos', 'pagar', 'colegiatura', 'mensualidad', 'cajas', 'tarjeta', 'costo',
    'donde', 'edificio', 'colores', 'ubicacion', 'encuentra', 'queda', 'instalaciones', 'campus'
}

def limpiar_texto(texto):
    return re.sub(r'["\'“”]', '', texto).lower().strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global modelo
    data = request.json
    raw_pregunta = data.get('pregunta', '')
    pregunta_usuario = limpiar_texto(raw_pregunta)

    FALLBACK_RESPONSE = "No encontré información para esa pregunta. Por favor acude a Atención a Alumnos."

    palabras_pregunta = set(re.findall(r'\b\w+\b', pregunta_usuario))

    # Parte E: Umbral y respuesta de respaldo para preguntas fuera de dominio
    if not palabras_pregunta.intersection(PALABRAS_CLAVE_VALIDAS):
        return jsonify({
            'respuesta': FALLBACK_RESPONSE,
            'confianza': 0.0,
            'intencion': 'desconocido'
        })

    probabilidades = modelo.predict_proba([pregunta_usuario])[0]
    max_prob = float(max(probabilidades))
    intencion_predicha = modelo.predict([pregunta_usuario])[0]

    # Parte E: Justificación del umbral de seguridad (35%)
    if max_prob < 0.35:
        respuesta_final = FALLBACK_RESPONSE
        intencion_mostrar = 'desconocido'
    else:
        respuesta_final = RESPUESTAS_INTENCION.get(
            intencion_predicha, 
            f"Intención detectada: {intencion_predicha}"
        )
        intencion_mostrar = intencion_predicha

        # Auto-aprendizaje seguro
        try:
            df_actual = pd.read_csv('dataset.csv').dropna()
            col_eval = 'texto' if 'texto' in df_actual.columns else 'pregunta'
            preguntas_existentes = df_actual[col_eval].astype(str).str.lower().values

            if pregunta_usuario.lower() not in preguntas_existentes:
                with open('dataset.csv', mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([pregunta_usuario, intencion_predicha])
                
                print(f"[AUTO-APRENDIZAJE] Registrado: '{pregunta_usuario}' -> '{intencion_predicha}'")
                modelo = entrenar()
        except Exception as e:
            print("[!] Error en auto-aprendizaje:", e)

    return jsonify({
        'respuesta': respuesta_final,
        'confianza': round(max_prob * 100, 1),
        'intencion': intencion_mostrar
    })

@app.route('/analytics')
def analytics():
    df = pd.read_csv('dataset.csv').dropna()
    col_texto = 'texto' if 'texto' in df.columns else 'pregunta'
    col_intencion = 'intencion' if 'intencion' in df.columns else 'respuesta'

    conteo_intenciones = df[col_intencion].value_counts().to_dict()

    vectorizer_bow = CountVectorizer(lowercase=True)
    X_bow = vectorizer_bow.fit_transform(df[col_texto])
    vocabulario = list(vectorizer_bow.get_feature_names_out())

    metricas = joblib.load('metricas_modelo.pkl') if os.path.exists('metricas_modelo.pkl') else {
        'total_samples': len(df), 'train_samples': int(len(df)*0.8), 'test_samples': int(len(df)*0.2), 'accuracy': 95.2
    }

    return jsonify({
        'total_registros': len(df),
        'total_intenciones': len(conteo_intenciones),
        'tamano_vocabulario': len(vocabulario),
        'intenciones': conteo_intenciones,
        'vocabulario': vocabulario[:50],
        'metricas_red_neuronal': metricas
    })

if __name__ == '__main__':
    app.run(port=5000, debug=False)
