import PyPDF2
from collections import Counter
import matplotlib.pyplot as plt
import os
from pdf2image import convert_from_path
import pytesseract
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

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
    print(f"No se pudo cargar NLTK completamente: {e}")
    print("Usando alternativas básicas.")
    def word_tokenize(text):
        return text.lower().split()
    spanish_stopwords = set(['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 'pero', 'si', 'no', 'en', 'de', 'a', 'para', 'por', 'con', 'del', '2020', '2015', 'años', 'año'])

# Lista personalizada de palabras a excluir
palabras_a_excluir = {'salud', 'plan', 'estratégico', 'ver', 'más', 'p.', 'fig.', '2020', '2015', 'año', 'formato', 'fuentes', 'fuente', 'años', 'causas', 'numerico', 'datos', 'chile', 'edad', 'según', 'muerte', 'persona', 'numérico', 'personas', 'hombres', 'gráfico', 'total', 'base', 'atención', 'nombre', 'mujeres', 'código', 'información', 'artículo', 'causa', 'definición', 'número', '2019', 'cada','ser','ley','nacional','documento','acuerdo','identificación','caso','fecha','art','nivel','país','general','corresponde','uso','texto'}
todas_stopwords = spanish_stopwords.union(palabras_a_excluir)

# Inicializar el analizador de sentimientos
sia = SentimentIntensityAnalyzer()

def extract_text_from_pdf(pdf_path):
    """Extrae texto de un archivo PDF si es posible. Si no, usa OCR."""
    text = ""

    # Intentamos extraer texto usando PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error usando PyPDF2 en {pdf_path}: {e}")

    # Si no se extrajo texto, usamos OCR con pdf2image y pytesseract
    if not text.strip():
        print(f"Usando OCR para extraer texto de imágenes en {pdf_path}")
        images = convert_from_path(pdf_path)
        for image in images:
            text += pytesseract.image_to_string(image, lang='spa')  # Cambia 'spa' si el idioma es diferente

    return text

def preprocess_text(text):
    """Preprocesa el texto: tokenización y eliminación de stopwords."""
    tokens = word_tokenize(text.lower())
    return [word for word in tokens if word.isalnum() and word not in todas_stopwords and len(word) > 2]

def analyze_sentiment(text):
    """Analiza el sentimiento del texto utilizando VADER."""
    return sia.polarity_scores(text)

# Lista general para almacenar todas las palabras procesadas de todos los PDFs
all_processed_words = []
sentiment_scores = {"neg": 0, "neu": 0, "pos": 0, "compound": 0}
pdf_count = 0  # Para calcular promedios de sentimientos

# Ruta a la carpeta que contiene los archivos PDF
pdf_folder = 'pdf'

# Iterar sobre todos los archivos PDF en la carpeta
for file_name in os.listdir(pdf_folder):
    if file_name.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder, file_name)
        
        # Extraer texto del PDF o usar OCR si es necesario
        raw_text = extract_text_from_pdf(pdf_path)
        
        if raw_text:
            # Preprocesamiento del texto
            processed_words = preprocess_text(raw_text)
            
            # Agregar las palabras procesadas a la lista general
            all_processed_words.extend(processed_words)

            # Analizar sentimientos del texto
            sentiment = analyze_sentiment(raw_text)
            for key in sentiment_scores:
                sentiment_scores[key] += sentiment[key]
            pdf_count += 1

        else:
            print(f"No se pudo extraer texto del archivo {file_name}.")

# Calcular promedios de sentimientos
if pdf_count > 0:
    for key in sentiment_scores:
        sentiment_scores[key] /= pdf_count

# Crear el gráfico general de todas las palabras de todos los PDFs
if all_processed_words:
    general_word_freq = Counter(all_processed_words)
    top_general_words = general_word_freq.most_common(25)
    
    # Graficar las palabras más frecuentes en todos los PDFs
    plt.figure(figsize=(12, 6))
    plt.bar([word for word, _ in top_general_words], [count for _, count in top_general_words])
    plt.title('Top 25 palabras más frecuentes en los PDFs sobre la salud')
    plt.xlabel('Palabras')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Mostrar las palabras más frecuentes en consola
    print("\nPalabras más frecuentes en todos los PDFs:")
    for word, freq in top_general_words:
        print(f"{word}: {freq}")
    
    # Mostrar gráfico de análisis de sentimientos como barras
    labels = ['Negativo', 'Neutral', 'Positivo', 'Compuesto']
    sizes = [sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos'], sentiment_scores['compound']]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color=['red', 'gray', 'green', 'blue'])
    plt.title('Distribución de Sentimientos en los PDFs')
    plt.xlabel('Tipo de Sentimiento')
    plt.ylabel('Promedio de Sentimiento')
    plt.tight_layout()
    plt.show()

else:
    print("No se encontraron palabras procesadas para generar el gráfico general.")
