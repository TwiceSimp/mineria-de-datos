import PyPDF2
from collections import Counter
import matplotlib.pyplot as plt
import os
from pdf2image import convert_from_path
import pytesseract
from wordcloud import WordCloud
import mysql.connector
from nltk.sentiment import SentimentIntensityAnalyzer

# Cargar NLTK o usar alternativas básicas si no está disponible
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    spanish_stopwords = set(stopwords.words('spanish'))
    print("NLTK cargado exitosamente.")
except Exception as e:
    # Si NLTK no se puede cargar, utilizar un tokenizador básico y una lista limitada de stopwords
    print(f"No se pudo cargar NLTK completamente: {e}")
    print("Usando alternativas básicas.")
    def word_tokenize(text):
        return text.lower().split()
    spanish_stopwords = set(['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'si', 'no', 'en', 'de', 'a', 'para', 'por', 'con', 'del', '2020', '2015', 'años', 'año'])

# Lista personalizada de palabras que queremos excluir (stopwords específicas)
palabras_a_excluir = {'salud', 'plan', 'estratégico', 'ver', 'más', 'p.', 'fig.', '2020', '2015', 'año', 'formato', 
                      'fuentes', 'fuente', 'años', 'causas', 'numerico', 'datos', 'chile', 'edad', 'según', 'muerte', 
                      'persona', 'numérico', 'personas', 'hombres', 'gráfico', 'total', 'base', 'atención', 'nombre', 
                      'mujeres', 'código', 'información', 'artículo', 'causa', 'definición', 'número', '2019', 'cada',
                      'ser', 'ley', 'nacional', 'documento', 'acuerdo', 'identificación', 'caso', 'fecha', 'art', 
                      'nivel', 'país', 'general', 'corresponde', 'uso', 'texto'}

# Unir las stopwords comunes y personalizadas
todas_stopwords = spanish_stopwords.union(palabras_a_excluir)

# Inicializar el analizador de sentimientos VADER
sia = SentimentIntensityAnalyzer()

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un archivo PDF si es posible. Si no, usa OCR."""
    text = ""

    # Intentamos extraer texto usando PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()  # Extraer texto de cada página
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error usando PyPDF2 en {pdf_path}: {e}")

    # Si no se extrajo texto, usamos OCR para procesar las imágenes en el PDF
    if not text.strip():
        print(f"Usando OCR para extraer texto de imágenes en {pdf_path}")
        images = convert_from_path(pdf_path)
        for image in images:
            text += pytesseract.image_to_string(image, lang='spa')  # Usar OCR para convertir imágenes en texto

    return text

def preprocess_text(text):
    """Preprocesa el texto: tokenización y eliminación de stopwords."""
    tokens = word_tokenize(text.lower())  # Convertir texto a minúsculas y tokenizar
    return [word for word in tokens if word.isalnum() and word not in todas_stopwords and len(word) > 2]  # Filtrar palabras

def analyze_sentiment(text):
    """Analiza el sentimiento del texto usando VADER."""
    return sia.polarity_scores(text)

def connect_to_mysql():
    """Conecta a la base de datos MySQL."""
    return mysql.connector.connect(
        host="192.168.56.101",  # Dirección del servidor MySQL
        user="edinson",         # Nombre de usuario
        password="root",        # Contraseña
        database="salud"        # Nombre de la base de datos
    )

def insert_into_mysql(cursor, nombre_archivo, palabras_clave):
    """Inserta los datos en la tabla MySQL."""
    sql = "INSERT INTO saludPublica (nombre_archivo, palabras_clave) VALUES (%s, %s)"
    palabras_clave_str = ", ".join(f"{word}: {count}" for word, count in palabras_clave)  # Formato para almacenar las palabras clave
    values = (nombre_archivo, palabras_clave_str)
    cursor.execute(sql, values)  # Ejecutar la inserción en la base de datos

def process_pdf_files(folder_path):
    """Procesa todos los archivos PDF en la carpeta especificada y almacena resultados en la base de datos."""
    conn = connect_to_mysql()
    cursor = conn.cursor()

    # Lista general para almacenar todas las palabras procesadas de todos los PDFs
    all_processed_words = []
    sentiment_scores = {"neg": 0, "neu": 0, "pos": 0, "compound": 0}  # Inicializar acumuladores de sentimientos
    pdf_count = 0  # Contador de PDFs procesados

    # Iterar sobre los archivos en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):  # Solo procesar archivos PDF
            pdf_path = os.path.join(folder_path, filename)
            try:
                # Extraer texto del PDF
                raw_text = extract_text_from_pdf(pdf_path)
                if raw_text:
                    # Preprocesar el texto extraído
                    processed_words = preprocess_text(raw_text)
                    word_frequencies = Counter(processed_words)
                    top_words = word_frequencies.most_common(25)  # Obtener las 25 palabras más frecuentes

                    # Insertar en la base de datos
                    insert_into_mysql(cursor, filename, top_words)
                    conn.commit()

                    # Agregar las palabras procesadas a la lista general para análisis gráfico
                    all_processed_words.extend(processed_words)

                    # Analizar sentimiento del texto
                    sentiment = analyze_sentiment(raw_text)
                    for key in sentiment_scores:
                        sentiment_scores[key] += sentiment[key]  # Acumular resultados de sentimientos

                    pdf_count += 1
                    print(f"Archivo {filename} procesado y almacenado con éxito.")
                else:
                    print(f"No se pudo extraer texto del archivo {filename}.")
            except Exception as e:
                print(f"Error al procesar el archivo {filename}: {str(e)}")

    # Cerrar la conexión con la base de datos
    cursor.close()
    conn.close()

    # Calcular promedio de los sentimientos
    if pdf_count > 0:
        for key in sentiment_scores:
            sentiment_scores[key] /= pdf_count  # Obtener los promedios de sentimientos

    return all_processed_words, sentiment_scores

# Ruta a la carpeta con los archivos PDF
pdf_folder = 'pdf'

# Procesar los archivos y obtener las palabras procesadas y los puntajes de sentimiento
all_processed_words, sentiment_scores = process_pdf_files(pdf_folder)

# Crear los gráficos si se encontraron palabras procesadas
if all_processed_words:
    general_word_freq = Counter(all_processed_words)  # Contar frecuencias de palabras
    top_general_words = general_word_freq.most_common(25)  # Obtener las 25 palabras más frecuentes
    
    # Graficar las palabras más frecuentes en todos los PDFs
    plt.figure(figsize=(12, 6))
    plt.bar([word for word, _ in top_general_words], [count for _, count in top_general_words])  # Gráfico de barras
    plt.title('Top 25 palabras más frecuentes en los PDFs sobre la salud')
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Generar una nube de palabras (Word Cloud)
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=todas_stopwords).generate_from_frequencies(general_word_freq)

    # Mostrar la nube de palabras
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Ocultar los ejes
    plt.title("Nube de Palabras de los PDFs sobre la salud")
    plt.show()

    # Mostrar gráfico de análisis de sentimientos
    labels = ['Negativo', 'Neutral', 'Positivo', 'Compuesto']
    sizes = [sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos'], sentiment_scores['compound']]
    
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['red', 'gray', 'green', 'blue'])  # Gráfico circular
    plt.title('Distribución de Sentimientos en los PDFs')
    plt.show()

else:
    print("No se encontraron palabras procesadas para generar los graficos ")
