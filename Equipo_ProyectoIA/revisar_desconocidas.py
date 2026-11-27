"""Revisión de las preguntas que el asistente no entendió.

Uso (desde la carpeta Equipo_ProyectoIA):  python3 revisar_desconocidas.py
"""
import csv
import base_datos as bd

DATASET = "datos/dataset.csv"

def agregar_al_dataset(pregunta, intencion):
    """Agrega el ejemplo al dataset si no existe. Devuelve True si lo agregó."""
    with open(DATASET, "r", newline="", encoding="utf-8") as f:
        contenido = f.read()

    existentes = {fila["texto"].strip().lower() for fila in csv.DictReader(contenido.splitlines())}
    if pregunta.strip().lower() in existentes:
        return False

    with open(DATASET, "a", newline="", encoding="utf-8") as f:
        if contenido and not contenido.endswith("\n"):
            f.write("\n")
        csv.writer(f, lineterminator="\n").writerow([pregunta.strip(), intencion])
    return True

def main():
    bd.crear_tablas()
    bd.cargar_base_conocimiento()
    pendientes = bd.listar_pendientes()
    if not pendientes:
        print("No hay preguntas pendientes de revisar.")
        return

    intenciones = bd.listar_intenciones()
    agregadas = 0

    for posicion, (id_desc, fecha_hora, pregunta, probable, confianza) in enumerate(pendientes, 1):
        print(f"\n[{posicion}/{len(pendientes)}] {fecha_hora}")
        print(f"Pregunta: {pregunta}")
        if probable:
            print(f"Intención más probable: {probable} ({confianza}%)")
        for numero, nombre in enumerate(intenciones, 1):
            print(f"  {numero}. {nombre}")

        while True:
            opcion = input("Número = asignar | d = descartar (fuera de tema) | s = saltar | q = salir: ").strip().lower()
            if opcion in ("d", "s", "q"):
                break
            if opcion.isdecimal() and 1 <= int(opcion) <= len(intenciones):
                break
            print("Opción no válida.")

        if opcion == "q":
            break
        if opcion == "s":
            continue
        if opcion == "d":
            bd.marcar_revisada(id_desc)
            print("Descartada.")
            continue

        intencion = intenciones[int(opcion) - 1]
        if agregar_al_dataset(pregunta, intencion):
            agregadas += 1
        bd.marcar_revisada(id_desc, intencion)
        print(f"Asignada a '{intencion}'.")

    if agregadas:
        print(f"\nSe agregaron {agregadas} ejemplos a {DATASET}.")
        print("Para que el asistente aprenda: ejecuta 'python3 experimentos.py' y reinicia 'asistente.py'.")

if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nRevisión interrumpida.")
