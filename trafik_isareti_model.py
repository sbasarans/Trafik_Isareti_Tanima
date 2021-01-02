# Gerekli kütüphanelerin yüklenmesi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 43
cur_path = os.getcwd()

# Görüntüleri ve etiketleri yükleme
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Görüntü yükleme hatası")

# Listeyi numpy dizilerine çevirme
data = np.array(data)
labels = np.array(labels)

# Verinin şekli (39209 , 30 , 30 ,3) dür. Yani 30x30 piksel boyutunda 39.209 görüntü var ve son 3 verinin renkli
# görüntüler içerdiği anlamına gelmektedir.

print(data.shape, labels.shape)
# Verisetini test ve eğitim verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# y_train ve x_train de bulunan verileri sıcak kodlamaya dönüştürmek için to_catecorical yöntemini kullanıyoruz.
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# CNN Modeli oluşturuyoruz.
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu',
                 input_shape=X_train.shape[1:]))  # relu aktivasyon fonksiyonunu kullanıyoruz
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))  # maxpool katmanı
model.add(Dropout(rate=0.25))  # bırakma katmanı
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))  # yoğun katman. softmax fonksiyonu kullanıyoruz.

# Model derlemesi için Adam optimizer kullanıyoruz . Kayıp fonksiyonu "catecorical_crossentropy" dir. Çünkü
# kategorize etmemiz gereken birden fazla sınıf var.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15  # epok sayısını en optimize şekilde tutmak önemlidir. Overfittinge sebep olabilir.
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save(
    "sumeyyenin_modeli.h5")  # Modeli model.fit komutu ile eğittikten sonra 32 ve 64 boyutlarıyla denedim. 64
# filtreleri ile daha iyi performans gösterdi.

# 15 epok sonucunda doğruluk sabitimiz %95 oldu.


# Matplotlib kütüphanesi ile grafiği doğruluk ve kayıp fonksiyonu için çiziyoruz.

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')  # doğruluk değeri
plt.xlabel('epochs')  # epok
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')  # kayıp fonksiyonu
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# Modeli test veri kümesi ile test etme. Daha önce ayırdığımız test.csv klasöründe test verilerimiz bulunmakta.
# Pandas kütüphanesi ile görüntü yolunu ve etiketini çıkarıyoruz.

from sklearn.metrics import accuracy_score

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data = []

# modeli tahmin etmek için görüntüleri 30x30 piksel olarak yeniden boyutlandırıyorum. Daha sonra tüm görüntü verilerini içeren bir numpy dizisi oluşturuyorum.

for img in imgs:
    image = Image.open(img)
    image = image.resize((30, 30))
    data.append(np.array(image))

X_test = np.array(data)

pred = model.predict_classes(X_test)

# Test verisi ile doğruluk. sklearn.metrics ten doğruluk skorunu içeri aktardıkve modelimizin gerçek etiketleri nasıl
# tahmin ettiğini gözlemledik.
from sklearn.metrics import accuracy_score

print(accuracy_score(labels, pred))

# Modelimiz %95 doğrulukla çalışıyor

model.save('traffic_classifier.h5')
