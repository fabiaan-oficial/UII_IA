import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ==============================================================================
# PROYECTO: UII AI Assistant
# ARCHIVO PRINCIPAL: asistente.py
# INSTITUCIÓN: Instituto Irapuato
# ALUMNO: Christian Fabián Álvarez Chávez (7mo Semestre ISD)
# DOCENTE: Mtra. Eréndira González Tinoco
# ==============================================================================

# 1. DICCIONARIO Y BASE DE DATOS INICIAL (Más de 10 registros)
BASE_CONOCIMIENTO = {
    "preguntas": [
        "¿Dónde puedo consultar mi horario?",
        "¿A qué hora son mis clases?",
        "¿Dónde veo mi tira de materias?",
        "¿Cuándo suben mi calificación?",
        "¿Dónde reviso mis notas del parcial?",
        "¿Qué día del mes se paga la colegiatura?",
        "¿Cuándo se paga la reinscripción?",
        "en donde puedo pagar la colegiatura?",
        "en donde se puede pagar? en el edificio de colores?",
        "donde esta ubicado el edificio de colores?",
        "¿Dónde solicito una constancia de estudios?"
    ],
    "respuestas": [
        "Puedes consultar tu horario y carga académica en el portal universitario oficial: https://uii.edu.mx/portal",
        "Puedes consultar tu horario y carga académica en el portal universitario oficial: https://uii.edu.mx/portal",
        "Puedes consultar tu horario y carga académica en el portal universitario oficial: https://uii.edu.mx/portal",
        "Tus calificaciones las encuentras en el Módulo de Historial Académico dentro del portal UII.",
        "Tus calificaciones las encuentras en el Módulo de Historial Académico dentro del portal UII.",
        "Las colegiaturas se pagan los días 10 de cada mes. Si cae en fin de semana se recorre al siguiente día hábil.",
        "La reinscripción se paga los primeros días de cada mes.",
        "Puedes pagar en cajas en el edificio de colores o en el aula escolar mediante pago con tarjeta.",
        "Puedes pagar en cajas en el edificio de colores o en el aula escolar mediante pago con tarjeta.",
        "Es el edificio entrando al Instituto Irapuato justo en frente. Le llaman edificio de colores porque tiene muchos colores en sus espejos.",
        "Solicita tu constancia en la oficina de Servicios Escolares dentro del edificio central."
    ]
}

# Variable global del modelo entrenado
MODELO = None

def entrenar_modelo_inicial():
    """Función para procesar con Pandas, limpiar con .dropna() y entrenar con Sklearn."""
    global MODELO
    df = pd.DataFrame(BASE_CONOCIMIENTO).dropna()
    X_train = df['preguntas']
    y_train = df['respuestas']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)),
        ('clf', MultinomialNB(alpha=0.1))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, 'modelo_uii_assistant.pkl')
    MODELO = pipeline
    return pipeline

# FUNCIÓN 1: Mostrar categorías de la institución
def mostrar_categorias():
    print("\n" + "="*50)
    print("       CATEGORÍAS DE ATENCIÓN UII ASSISTANT")
    print("="*50)
    print(" 1. Horarios y Tira de Materias")
    print(" 2. Calificaciones y Evaluaciones Parciales")
    print(" 3. Pagos de Colegiatura y Reinscripción")
    print(" 4. Ubicación de Edificios y Servicios Escolares")
    print("="*50 + "\n")

# FUNCIÓN 2: Consultar información mediante Machine Learning + Fallback
def consultar_informacion():
    global MODELO
    if MODELO is None:
        MODELO = entrenar_modelo_inicial()

    FALLBACK_RESPONSE = "No encontré información para esa pregunta. Por favor acude a Atención a Alumnos."
    UMBRAL_CONFIANZA = 0.25

    print("\n--- CONSULTA DE INFORMACIÓN INSTITUCIONAL ---")
    pregunta_usuario = input("Escribe tu pregunta sobre la UII: ").strip()

    if not pregunta_usuario:
        print(" [!] No ingresaste ninguna pregunta.")
        return

    # Predicción y cálculo de certeza
    probabilidades = MODELO.predict_proba([pregunta_usuario])[0]
    max_prob = float(max(probabilidades))
    prediccion = MODELO.predict([pregunta_usuario])[0]

    print("\n" + "-"*50)
    if max_prob < UMBRAL_CONFIANZA:
        print(f"[Asistente UII]: {FALLBACK_RESPONSE}")
    else:
        print(f"[Asistente UII]: {prediccion}")
    
    print(f"[Nivel de Confianza del Modelo]: {round(max_prob * 100, 2)}%")
    print("-" * 50 + "\n")

# FUNCIÓN 3: Menú principal interactivo con ciclo While
def menu_principal():
    entrenar_modelo_inicial()
    
    opcion = ""
    while opcion != "3":
        print("\n==================================================")
        print("         SISTEMA ASISTENTE IA - INSTITUTO IRAPUATO")
        print("==================================================")
        print(" 1. Consultar información")
        print(" 2. Mostrar categorías")
        print(" 3. Salir")
        print("==================================================")
        
        opcion = input("Selecciona una opción (1-3): ").strip()

        if opcion == "1":
            consultar_informacion()
        elif opcion == "2":
            mostrar_categorias()
        elif opcion == "3":
            print("\n¡Gracias por utilizar el UII AI Assistant! Hasta luego.\n")
        else:
            print("\n [!] Opción no válida. Intenta de nuevo con 1, 2 o 3.")

if __name__ == '__main__':
    menu_principal()
