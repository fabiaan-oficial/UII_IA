import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ==============================================================================
# PROYECTO: UII AI Assistant
# SCRIPT: preparar_dataset.py
# FASE: Carga, Normalización y Bag of Words (CountVectorizer)
# INSTITUCIÓN: Instituto Irapuato
# ==============================================================================

def preparar_y_analizar_dataset():
    print("==========================================================")
    print(" 1. CARGA Y VALIDACIÓN DEL ARCHIVO CSV")
    print("==========================================================")
    
    archivo_csv = 'dataset.csv'
    try:
        df = pd.read_csv(archivo_csv)
        print(f"[✓] Archivo '{archivo_csv}' cargado correctamente.")
    except Exception as e:
        print(f"[!] Error al cargar el archivo CSV: {e}")
        return

    # 2. Comprobar que existan las columnas requeridas ('texto' e 'intencion')
    columnas_requeridas = ['texto', 'intencion']
    for col in columnas_requeridas:
        if col in df.columns:
            print(f"[✓] Columna requerida '{col}' encontrada.")
        else:
            print(f"[!] ERROR: La columna '{col}' NO existe en el CSV.")
            return

    # Eliminar registros vacíos o nulos
    df = df.dropna()
    print(f"[✓] Total de registros válidos y limpios: {len(df)}")

    print("\n==========================================================")
    print(" 2. REVISIÓN DE CANTIDAD DE REGISTROS POR INTENCIÓN")
    print("==========================================================")
    conteo_intenciones = df['intencion'].value_counts()
    print(conteo_intenciones)
    print(f"\nTotal de intenciones detectadas: {len(conteo_intenciones)}")

    print("\n==========================================================")
    print(" 3. DEMOSTRACIÓN BAG OF WORDS (CountVectorizer)")
    print("==========================================================")
    
    X_texto = df['texto']
    
    # Inicializar CountVectorizer para Bag of Words
    vectorizer = CountVectorizer(lowercase=True)
    
    # Transformar el texto a vectores numéricos
    X_bow = vectorizer.fit_transform(X_texto)
    vocabulario = vectorizer.get_feature_names_out()

    print(f"[✓] Transformación a Bag of Words realizada exitosamente.")
    print(f"-> Tamaño del vocabulario único extraído: {len(vocabulario)} palabras.")
    print(f"-> Forma de la matriz resultante (Registros, Palabras): {X_bow.shape}")

    print("\n--- EJEMPLO DE MUESTRA DE PALABRAS EN VOCABULARIO ---")
    print(vocabulario[:20])  # Muestra las primeras 20 palabras extraídas

    print("\n--- EJEMPLO DE VECTORIZACIÓN DE LA PRIMERA FRASE ---")
    print(f"Frase original: '{X_texto.iloc[0]}'")
    print(f"Vector numérico (frecuencias distintas de cero):")
    print(X_bow[0].toarray())
    print("==========================================================")

if __name__ == '__main__':
    preparar_y_analizar_dataset()
