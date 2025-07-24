import os
import cv2
import shutil
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
import subprocess

# === Configuración ===
RAW_DIR = '../raw_dataset'
PREPROCESS_DIR = '../preprocessed_dataset'
OUTPUT_DIR = '../dataset'
CSV_PATH = '../labels_deepface.csv'
IMG_SIZE = (48, 48)
RAR_PATH = 'C:\\Program Files\\WinRAR\\WinRAR.exe'

def compress_to_rar(folder_path, rar_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    rar_file = os.path.join(parent_dir, f"{rar_name}.rar")

    try:
        subprocess.run([RAR_PATH, 'a', '-r', rar_file, folder_path], check=True)
        print(f"Carpeta '{folder_path}' comprimida en '{rar_file}'")
    except Exception as e:
        print(f"Error al comprimir: {e}")

def preprocess_images():
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    print("Preprocesando imágenes...")

    for filename in tqdm(os.listdir(RAW_DIR)):
        path = os.path.join(RAW_DIR, filename)
        if not filename.lower().endswith('.jpg'):
            continue
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            cv2.imwrite(os.path.join(PREPROCESS_DIR, filename), img)
        except Exception as e:
            print(f"Error procesando {filename}: {e}")

def auto_label_images():
    labels = []

    print("Etiquetando con DeepFace...")
    for filename in tqdm(os.listdir(PREPROCESS_DIR)):
        img_path = os.path.join(PREPROCESS_DIR, filename)
        if not filename.endswith('.jpg'):
            continue

        try:
            result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
            label = result[0]['dominant_emotion']
        except Exception as e:
            print(f"Error analizando {filename}: {e}")
            label = 'unknown'

        labels.append({'filename': filename, 'label': label})

    df = pd.DataFrame(labels)
    df.to_csv(CSV_PATH, index=False)
    print(f"CSV guardado en {CSV_PATH}")
    return df

def organize_images(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Organizando imágenes por emoción...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        label = row['label']
        if label == 'unknown':
            continue

        label_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(label_dir, exist_ok=True)

        src = os.path.join(PREPROCESS_DIR, row['filename'])
        dst = os.path.join(label_dir, row['filename'])

        if os.path.exists(src):
            shutil.copyfile(src, dst)

    compress_to_rar(OUTPUT_DIR, 'dataset')

# === Ejecución ===
if __name__ == "__main__":
    preprocess_images()
    df = auto_label_images()
    organize_images(df)
    print("Todo listo con DeepFace!")