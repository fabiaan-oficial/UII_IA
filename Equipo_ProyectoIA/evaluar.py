"""Examen del asistente con preguntas que NO se usan para entrenar (pruebas/examen.csv).
No guarda nada en la base de datos.  Uso:  python3 evaluar.py
"""
import csv
from modelo import ModeloAsistente

def main():
    ia = ModeloAsistente()
    with open("pruebas/examen.csv", "r", newline="", encoding="utf-8") as f:
        preguntas = list(csv.DictReader(f))

    correctas, incoherentes, sin_respuesta, fallos = 0, 0, 0, []
    for fila in preguntas:
        intencion, confianza, es_desconocida = ia.predecir(fila["texto"])
        obtenida = "fuera_de_alcance" if es_desconocida else intencion
        if obtenida == fila["intencion"]:
            correctas += 1
            continue
        if obtenida == "fuera_de_alcance":
            sin_respuesta += 1
        else:
            incoherentes += 1
        fallos.append((fila["texto"], fila["intencion"], obtenida, confianza))

    for texto, esperada, obtenida, confianza in fallos:
        print(f"FALLO: '{texto}' | esperaba {esperada} | dio {obtenida} ({confianza:.1f}%)")
    print(f"\nCorrectas: {correctas}/{len(preguntas)}")
    print(f"Respuestas equivocadas: {incoherentes}")
    print(f"Preguntas válidas sin contestar: {sin_respuesta}")

if __name__ == "__main__":
    main()
