import base_datos as bd
from modelo import ModeloAsistente

bd.crear_tablas()
bd.cargar_base_conocimiento("datos/respuestas.json")
ia = ModeloAsistente(umbral_confianza=0.60)

pruebas = [
    # 15 consultas DENTRO de alcance
    ("Hola, buenos días", "Dentro"),
    ("Qué tal, buenas tardes", "Dentro"),
    ("¿Qué becas ofrece la universidad?", "Dentro"),
    ("Requisitos para solicitar una beca académica", "Dentro"),
    ("¿Cuándo son las inscripciones?", "Dentro"),
    ("¿Qué documentos necesito para reinscribirme?", "Dentro"),
    ("¿Dónde puedo consultar mi horario de clases?", "Dentro"),
    ("Quiero ver mi tira de materias", "Dentro"),
    ("Necesito tramitar una constancia de estudios", "Dentro"),
    ("¿Cómo pido mi kárdex oficial?", "Dentro"),
    ("¿Dónde se paga la colegiatura?", "Dentro"),
    ("Métodos de pago para trámites escolares", "Dentro"),
    ("¿Cuál es la dirección del campus?", "Dentro"),
    ("¿Dónde está Control Escolar?", "Dentro"),
    ("¿A qué hora abren caja para pagar?", "Dentro"),
    # 5 consultas FUERA de alcance
    ("¿A qué hora sale el camión para León?", "Fuera"),
    ("¿Cuál es la receta para hacer tacos al pastor?", "Fuera"),
    ("¿Quién ganó el partido de fútbol ayer?", "Fuera"),
    ("¿Me recomiendas una película de terror?", "Fuera"),
    ("¿Cuánto cuesta un boleto de avión a Cancún?", "Fuera"),
]

print("\n=========================================================================================================")
print(f"{'#':<3} | {'Tipo':<7} | {'Consulta':<46} | {'Intención Prob.':<19} | {'Confianza':<9} | {'Estado'}")
print("=" * 105)

for idx, (pregunta, tipo) in enumerate(pruebas, 1):
    intencion_prob, confianza, es_desconocida = ia.predecir(pregunta)
    estado = "DESCONOCIDA (CSV)" if es_desconocida else "ACEPTADA"
    bd.registrar_consulta(pregunta, "desconocido" if es_desconocida else intencion_prob, confianza)
    if es_desconocida:
        bd.registrar_desconocida(pregunta, intencion_prob, confianza)

    print(f"{idx:<3} | {tipo:<7} | {pregunta:<46} | {intencion_prob:<19} | {confianza:>7.2f}% | {estado}")

print("=" * 105)
print("\nContenido generado en 'desconocidas.csv':\n")
with open("desconocidas.csv", "r", encoding="utf-8") as f:
    print(f.read())
