import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

PATH = r'C:\Users\moldo\Desktop\Proiect_bmw'
CATEGORII = ['m5.e28', 'm5.e34', 'm5.e39', 'm5.e60', 'm5.f10', 'm5.f90', 'm5.g90']
IMG_SIZE = 224
BATCH = 8

data, labels = [], []
poze_per_folder = {}


print("📥 Se citesc pozele...")
for i, cat in enumerate(CATEGORII):
    folder = os.path.join(PATH, cat)
    total_poze = 0
    if not os.path.exists(folder):
        print(f"⚠️ Folderul {cat} nu există!")
        poze_per_folder[cat] = 0
        continue
    for img_name in os.listdir(folder):
        total_poze += 1
        try:
            img = cv2.imread(os.path.join(folder, img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(i)
        except: 
            pass
    poze_per_folder[cat] = total_poze
    print(f"✅ {cat}: {total_poze} poze găsite și încărcate.")

X = np.array(data) / 255.0
y = np.array(labels)

print(f"\n📦 Total poze încărcate: {len(X)}")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"📊 Poze train/test per clasă:")
for i, cat in enumerate(CATEGORII):
    total = np.sum(y == i)
    train = np.sum(y_train == i)
    test = np.sum(y_test == i)
    print(f"{cat:<10} Total: {total:<3} | Train: {train:<3} | Test: {test:<3}")


datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(CATEGORII), activation='softmax')
])

model.compile(optimizer=optimizers.Adam(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\n🚀 Începe antrenarea...")
model.fit(datagen.flow(X_train, y_train, batch_size=BATCH),
          epochs=25,
          validation_data=(X_test, y_test),
          verbose=1)


loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 ACURATEȚE TOTALĂ: {acc*100:.2f}%\n")

y_pred = np.argmax(model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)

print("📊 ACURATEȚEA PE FIECARE CLASĂ:")
print("="*80)
print(f"{'Clasa':<10} {'Poze totale':<12} {'Corecte/Total Test':<20} {'Acuratețe (%)':<15}")
print("-"*80)

for i, cat in enumerate(CATEGORII):
    total_test = np.sum(y_test == i)
    corecte = cm[i, i] if total_test > 0 else 0
    acc_clasa = (corecte / total_test * 100) if total_test > 0 else 0
    print(f"{cat:<10} {poze_per_folder[cat]:<12} {corecte}/{total_test:<20} {acc_clasa:<15.1f}")

print("="*80)