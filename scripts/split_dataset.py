import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

input_dir = '../dataset'
output_base = '../dataset_split'

# Crear carpetas de salida
splits = ['train', 'validation', 'test']
for split in splits:
    for emotion in os.listdir(input_dir):
        os.makedirs(os.path.join(output_base, split, emotion), exist_ok=True)

# Dividir imágenes por clase
for emotion in os.listdir(input_dir):
    emotion_dir = os.path.join(input_dir, emotion)
    if not os.path.isdir(emotion_dir):
        continue

    images = os.listdir(emotion_dir)
    if len(images) < 10:
        print(f"Muy pocas imágenes para la clase {emotion}, se omite.")
        continue

    # División 70-10-20
    train_imgs, temp_imgs = train_test_split(images, train_size=0.7, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=2/3, random_state=42)  # test ≈ 20%

    # Copiar archivos
    for img_list, split in zip([train_imgs, val_imgs, test_imgs], splits):
        for img in img_list:
            src = os.path.join(emotion_dir, img)
            dst = os.path.join(output_base, split, emotion, img)
            shutil.copyfile(src, dst)

print("División completada exitosamente.")

print("\nResumen por conjunto:")
for split in splits:
    print(f"\n{split.upper()}")
    for emotion in os.listdir(os.path.join(output_base, split)):
        count = len(os.listdir(os.path.join(output_base, split, emotion)))
        print(f"{emotion}: {count}")