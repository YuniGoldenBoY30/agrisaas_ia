import os

def encontrar_png_sin_json():
    # Obtener la ruta del directorio actual
    carpeta = os.path.dirname(os.path.abspath(__file__))
    
    # Listar archivos .png y .json en la carpeta
    archivos_png = [f for f in os.listdir(carpeta) if f.endswith('.png')]
    archivos_json = [f for f in os.listdir(carpeta) if f.endswith('.json')]
    
    # Extraer los nombres base (sin extensión)
    nombres_png = {os.path.splitext(f)[0] for f in archivos_png}
    nombres_json = {os.path.splitext(f)[0] for f in archivos_json}
    
    # Encontrar png sin json
    sin_json = nombres_png - nombres_json
    
    return sorted(sin_json)

def main():
    print("Buscando archivos PNG sin archivo JSON correspondiente...")
    sin_json = encontrar_png_sin_json()
    
    if sin_json:
        print("Archivos PNG sin JSON:", ", ".join(sin_json))
    else:
        print("Todos los archivos PNG tienen su archivo JSON correspondiente.")

if __name__ == "__main__":
    main()