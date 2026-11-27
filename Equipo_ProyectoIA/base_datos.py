import os
import csv
import sqlite3
import json
from datetime import datetime

DB_NAME = "asistente.db"
CSV_DESCONOCIDAS = "desconocidas.csv"

def conectar():
    return sqlite3.connect(DB_NAME)

def crear_tablas():
    conn = conectar()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS intenciones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT UNIQUE NOT NULL,
            respuesta TEXT NOT NULL,
            fuente TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consultas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_hora TEXT NOT NULL,
            pregunta TEXT NOT NULL,
            intencion_predicha TEXT NOT NULL,
            confianza REAL NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS desconocidas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha_hora TEXT NOT NULL,
            pregunta TEXT NOT NULL,
            revisada INTEGER DEFAULT 0,
            intencion_probable TEXT,
            confianza REAL,
            intencion_asignada TEXT
        )
    """)

    # Actualizar bases creadas con la versión anterior (sin borrar datos)
    cursor.execute("PRAGMA table_info(desconocidas)")
    columnas = {fila[1] for fila in cursor.fetchall()}
    for columna, tipo in [("intencion_probable", "TEXT"), ("confianza", "REAL"), ("intencion_asignada", "TEXT")]:
        if columna not in columnas:
            cursor.execute(f"ALTER TABLE desconocidas ADD COLUMN {columna} {tipo}")

    conn.commit()
    conn.close()

    if not os.path.exists(CSV_DESCONOCIDAS):
        with open(CSV_DESCONOCIDAS, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fecha", "hora", "pregunta", "intencion_mas_probable", "confianza"])

    completar_desde_csv()

def completar_desde_csv():
    """Rellena intención y confianza de registros antiguos usando desconocidas.csv."""
    with open(CSV_DESCONOCIDAS, "r", newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))

    conn = conectar()
    for fila in filas:
        conn.execute("""
            UPDATE desconocidas SET intencion_probable = ?, confianza = ?
            WHERE fecha_hora = ? AND pregunta = ? AND intencion_probable IS NULL
        """, (fila["intencion_mas_probable"], float(fila["confianza"]),
              f"{fila['fecha']} {fila['hora']}", fila["pregunta"]))
    conn.commit()
    conn.close()

def cargar_base_conocimiento(ruta_json="datos/respuestas.json"):
    with open(ruta_json, "r", encoding="utf-8") as archivo:
        datos = json.load(archivo)

    conn = conectar()
    cursor = conn.cursor()

    for nombre, info in datos.items():
        cursor.execute("""
            INSERT OR REPLACE INTO intenciones (nombre, respuesta, fuente)
            VALUES (?, ?, ?)
        """, (nombre, info["respuesta"], info["fuente"]))

    conn.commit()
    conn.close()

def obtener_respuesta_bd(intencion):
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("SELECT respuesta, fuente FROM intenciones WHERE nombre = ?", (intencion,))
    fila = cursor.fetchone()
    conn.close()
    return fila

def registrar_consulta(pregunta, intencion_predicha, confianza):
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO consultas (fecha_hora, pregunta, intencion_predicha, confianza)
        VALUES (?, ?, ?, ?)
    """, (fecha_hora, pregunta, intencion_predicha, round(confianza, 2)))
    conn.commit()
    conn.close()

def registrar_desconocida(pregunta, intencion_mas_probable, confianza):
    ahora = datetime.now()
    fecha = ahora.strftime("%Y-%m-%d")
    hora = ahora.strftime("%H:%M:%S")
    fecha_hora = f"{fecha} {hora}"

    # Guardar en SQLite
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO desconocidas (fecha_hora, pregunta, revisada, intencion_probable, confianza)
        VALUES (?, ?, 0, ?, ?)
    """, (fecha_hora, pregunta, intencion_mas_probable, round(confianza, 2)))
    conn.commit()
    conn.close()

    # Guardar en desconocidas.csv (Paso 4)
    with open(CSV_DESCONOCIDAS, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([fecha, hora, pregunta, intencion_mas_probable, round(confianza, 2)])

def listar_pendientes():
    conn = conectar()
    filas = conn.execute("""
        SELECT id, fecha_hora, pregunta, intencion_probable, confianza
        FROM desconocidas WHERE revisada = 0 ORDER BY id
    """).fetchall()
    conn.close()
    return filas

def listar_intenciones():
    conn = conectar()
    filas = conn.execute("SELECT nombre FROM intenciones ORDER BY nombre").fetchall()
    conn.close()
    return [fila[0] for fila in filas]

def marcar_revisada(id_desconocida, intencion_asignada=None):
    """intencion_asignada=None significa que la pregunta se descartó."""
    conn = conectar()
    conn.execute(
        "UPDATE desconocidas SET revisada = 1, intencion_asignada = ? WHERE id = ?",
        (intencion_asignada, id_desconocida)
    )
    conn.commit()
    conn.close()
