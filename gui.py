#Tkinter ile trafik işaretleri sınıflandırıcımız için bir grafik kullanıcı arayüzü oluşturacağım. Tkinter standart python kütüphanesindeki bir gui araç setidir.
#Gerekli kütüphaneleri yükleme
import tkinter as tk
from tkinter import filedialog

from tkinter import *
from PIL import ImageTk, Image
#import PIL
import tkinter

import numpy
#Keras ile eğittiğimiz modeli yükledik. Görüntüyü yüklemek için GUİ oluşturuyoruz.
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
model = load_model("traffic_classifier.h5")

#Sınıflandırmak için etiketler sözlüğü
classes = { 1:'Hız limiti (20km/h)',
            2:'Hız limiti (30km/h)',
            3:'Hız limiti (50km/h)',
            4:'Hız limiti(60km/h)',
            5:'Hız limiti (70km/h)',
            6:'Hız limiti (80km/h)',
            7:'Hız limiti Sonu (80km/h)',
            8:'Hız limiti (100km/h)',
            9:'Hız limiti (120km/h)',
           10:'Geçiş yok',
           11:'3.5 tondan fazla araç geçişi yasak',
           12:'Kavşakta geçiş hakkı',
           13:'Öncelikli yol',
           14:'Yol ver işareti',
           15:'Dur',
           16:'Araç girşi yasak',
           17:'3.5 tondan fazla araç geçişi yasak',
           18:'Giriş yok',
           19:'Genel uyarı',
           20:'Tehlikeli sola dönüş',
           21:'Tehlikeli sağa dönüş',
           22:'Çift Kasis',
           23:'Engebeli yol',
           24:'Kaygan yol',
           25:'Sağda daralan yol',
           26:'Yol çalışması',
           27:'Traffik sinyali',
           28:'Yayalar',
           29:'Çocuk geçişi',
           30:'Bisiklet geçişi',
           31:'Dikkat donma tehlikesi',
           32:'Vahşi hayvan çıkabilir',
           33:'Son hız geçiş sınırı',
           34:'Sağa dönün',
           35:'Sola dönün',
           36:'İleri',
           37:'Düz veya sağa dön',
           38:'Düz veya sola dön',
           39:'Sağdan devam et',
           40:'Soldan devam edin',
           41:'Döner kavşak zorunlu',
           42:'Geçiş sonu',
           43:'Geçemeyen araçların sonu > 3.5 tons' }
                 
# GUI içe aktarma
top=tk.Tk()
top.geometry('800x600')
top.title('Trafik işareti Sınıflandırıcısı')
top.configure(background='#CDCDCD')

label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

#Classify işlevi görüntüyü şekil boyutuna dönüştürür (1,30,30,3). Çünkü trafik işaretini tahmin etmek için modeli oluştururken kullandığımız aynı boyutu sağlamamız gerekir.
#Ardından model.predict.classes bize ait olduğu sınıfı temsil eden 0 ile 42 arası bir sayı döndürür . Sınıfla ilgili bilgileri almak için yukarıda oluşturduğumuz sözlüğü kullanıyoruz.
# Sonra sınıfı tahmin ediyoruz

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict_classes([image])[0]
    sign = classes[pred+1]
    print(sign)
    label.configure(foreground='#011638', text=sign) 
   

def show_classify_button(file_path):
    classify_b=Button(top,text="Görüntüyü Tanımla",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Görüntü yükleyin",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Trafik işaretini öğren ",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()
