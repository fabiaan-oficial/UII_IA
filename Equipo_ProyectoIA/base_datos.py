import sqlite3
import json
from datetime import datetime

DB_NAME = "asistente.db"

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
            revisada INTEGER DEFAULT 0
        )
    """)

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

def registrar_desconocida(pregunta):
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO desconocidas (fecha_hora, pregunta, revisada)
        VALUES (?, ?, 0)
    """, (fecha_hora, pregunta))
    conn.commit()
    conn.close()
