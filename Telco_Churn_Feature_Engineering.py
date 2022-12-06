


#              TELCO CHURN FEATURE ENGINEERING

"""
            İş Problemi

    Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
    geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi
    ve özellik mühendisliği adımlarını gerçekleştirmeniz beklenmektedir.

            Veri Seti Hikayesi
    Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve
    İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.
    Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

    CustomerId       : Müşteri İd’si
    Gender           : Cinsiyet
    SeniorCitizen    : Müşterinin yaşlı olup olmadığı (1, 0)
    Partner          : Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
    Dependents       : Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır)
    tenure           : Müşterinin şirkette kaldığı ay sayısı
    PhoneService     : Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
    MultipleLines    : Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
    InternetService  : Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
    OnlineSecurity   : Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
    OnlineBackup     : Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
    DeviceProtection : Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
    TechSupport      : Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
    StreamingTV      : Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
    StreamingMovies  : Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
    Contract         : Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
    PaperlessBilling : Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
    PaymentMethod    : Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
    MonthlyCharges   : Müşteriden aylık olarak tahsil edilen tutar
    TotalCharges     : Müşteriden tahsil edilen toplam tutar
    Churn            : Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

"""

####################################################################
#       Görev 1 : Keşifçi Veri Analizi
####################################################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor #cok degiskenli aykırı deger yakalama yöntemi(LOF)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler #standartlastırma, dönüştürme fonksiyonları

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

#----------------Adım 1: Genel resmi inceleyiniz.-----------------------

#her seferinde tekrar tekrar veri okuma islemi yapılmaması icin:
def load():
    data = pd.read_csv("datasets/Telco-Customer-Churn.csv")
    return data


df = load()
df.head()
df.shape
df.info()


#degiskenlerin isimlerini büyütüyoruz,df in sütunlarında gez,yakaladıgın ismi büyült
df.columns = [col.upper() for col in df.columns]

#esik deger hesaplama fonk
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75): #q1 ve q3ü 5e 95 ya da 1e 99 da alabilirsin
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

#aykırı deger var mı yok mu fonk
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)  #q1 ve q3 ü burada yazarsak farklı deger icin,yukarıda col_name'in yanına da eklememiz lazım
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False



#--------------Adım 2: Numerik ve kategorik değişkenleri yakalayınız.------------------------

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat= grab_col_names(df)
#cıktı:Variables: 21=cat_cols: 17 + num_cols: 2 + cat_but_car: 2
cat_cols
num_cols
cat_but_car
num_but_cat

cat_but_car = [col for col in cat_but_car if col not in "CUSTOMERID"]

#TOTALCHARGES object görünümlü nümerik degisken
#df["TOTALCHARGES"] = df["TOTALCHARGES"].astype("float64") ???? bu neden olmadı
df["TOTALCHARGES"] = pd.to_numeric(df["TOTALCHARGES"], errors="coerce")  #?? errors="coerce" eklemeyince neden olmuyor
num_cols= num_cols + cat_but_car

"""ya da 
df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
df['TotalCharges'] = df['TotalCharges'].astype("float64")
"""
df["TOTALCHARGES"].dtypes

#---------Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.-----------------

#NÜMERİK

df[num_cols].describe().T

#nümerik kolonlara aykırı deger var mı diye soralım
for col in num_cols:
    print(col, check_outlier(df, col))

#num_summary:bi sayısal degiskenin ceyreklik degerlerini göstermek ve garfigini olusturmak
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


#KATEGORİK

#kategorik degiskenlerin sınıflarına ve bu sınıfların oranlarına bakalım
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe) #gorsellestirmek istiyorsak yukarıdaki "plot=False"u True yaparız
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

#Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması,
# hedef değişkene göre numerik değişkenlerin ortalaması)

#Kategorik değişkenlere göre hedef değişkenin ortalaması
#hedef degiskenim:CHURN kategorik bir degisken,
df["CHURN"].head()


df.loc[df["CHURN"]=="Yes", "CHURN"] = 1
df.loc[df["CHURN"]=="No", "CHURN"] = 0
#ya da df["CHURN"] = df["CHURN"].apply(lambda x : 1 if x == "Yes" else 0)
df["CHURN"].head()

cat_cols
cat_cols= [col for col in cat_cols if col not in "CHURN"] #CHURN kategoriden cıkardım,grupby'a alacagım icin

"""df.groupby("GENDER").agg({"CHURN": "mean"})
df.groupby("CONTRACT").agg({"CHURN": "mean"})
#tek tek degilde bütün cat_cols larda yapalım
for col in cat_cols:
    print(df.groupby(col).agg({"CHURN": "mean"}))
"""

def target_summary_with_cat(dataframe, target, cat_cols):
    print(cat_cols)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(cat_cols)[target].mean(),
                        "Count": dataframe[cat_cols].value_counts(),
                        "Ratio": 100 * dataframe[cat_cols].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "CHURN", col)

# hedef değişkene göre numerik değişkenlerin ortalaması
num_cols
"""df.groupby("CHURN").agg({"TENURE": "mean"})

for col in num_cols:
    print(df.groupby("CHURN").agg({col: "mean"}))
  """

def target_summary_with_num(dataframe, target, num_cols):
    print(dataframe.groupby(target).agg({num_cols: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "CHURN", col)



#-------------------Adım 5: Aykırı gözlem analizi yapınız.-------------------------------

#yukarıda aykırı deger olmadıgını gözlemlemiştik
check_outlier(df, num_cols)

#eger aykırı deger olsaydı bu fonks ile bulabilirdik

def grab_outliers(dataframe, col_name, index=False): #index=False ilgili outliersların indexini istemiyoruz eger istersek True yazarız
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10: #shape[0]:gözlem sayısı var, shape[1]:degisken sayısı var
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head()) #degiskendeki gözlem sayısı 10dan fazlaysa hepsini getirme,ilk besi göster
    else: #degiskendeki gözlem sayısı 10dan az ise hepsini getir göreyim
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "TENURE")




#Adım 6: Eksik gözlem analizi yapınız.


# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()
df.info()
# degiskenlerdeki eksik deger sayisi
df.isnull().sum()
# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

#TOTALCHARGES 11 tane eksik deger gözlemliyoruz,fakat TOTALCHARGES'ı nümerik yapmadan önce eksik deger görünmüyordu
#eksik deger indexlerini bulup,kategorik oldugu df'e gittim ve burada ne yazdıgına baktım
#df["TotalCharges"][6754]  cıktısı su sekildeydi: ' '  yani bosluk konulmuş bu da eksik deger olarak algılanmamış
#TOTALCHARGES'ı eksik olan müsterilerin hepsinin CHURN ve TENURE sıfır(0), yani bunlar yeni müsteriler

#Adım 7: Korelasyon analizi yapınız.

df[num_cols].corr()
sns.heatmap(df[num_cols].corr())
plt.show()

#???df.corrwith(df["CHURN"]).sort_values(ascending=False)???


################################################################
#           Görev 2 : Feature Engineering
################################################################

#Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

#Aykırı deger bulamamıştık,bir işlem  yapmıyoruz

#TOTALCHARGES 11 eksik degerimiz vardı,bunları silebiliriz ya da doldurabiliriz. ben 0 ile doldurmak istiyorum
dff= load()
dff.columns = [col.upper() for col in dff.columns]
dff[dff.isnull().any(axis=1)] #işlem yapılmamıs yani TOTALCHARGES kategorik iken

df[df.isnull().any(axis=1)]

df["TOTALCHARGES"].fillna(0, inplace=True) #medyan ile de doldurabilirsin



#-----------------------Adım 2: Yeni değişkenler oluşturunuz.-----------------------------

# Tenure değişkeninden yıllık değişken oluşturma
df.loc[(df["TENURE"]>=0) & (df["TENURE"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["TENURE"]>12) & (df["TENURE"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["TENURE"]>24) & (df["TENURE"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["TENURE"]>36) & (df["TENURE"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["TENURE"]>48) & (df["TENURE"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["TENURE"]>60) & (df["TENURE"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

# Kontratı 1 veya 2 yıllık müşterileri birleştirme
df["NEW_CONTRACT"] = df["CONTRACT"].apply(lambda x: 1 if x in ["One year","Two year"] else 0)

# Herhangi bir destek, yedek veya koruma alan kişiler
df["NEW_BACKUP_PROTECTION_SUPPORT"] = df.apply(lambda x: 1 if (x["ONLINEBACKUP"] == "Yes") or (x["DEVICEPROTECTION"] == "Yes") or (x["TECHSUPPORT"] == "Yes") else 0, axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_YOUNG_MONTH_TO_MONTH"] = df.apply(lambda x: 1 if (x["NEW_CONTRACT"] == 0) and (x["SENIORCITIZEN"] == 0) else 0, axis=1)

# Kişinin toplam aldığı servis sayısı
df["NEW_TOTALSERVICE"] = (df[["PHONESERVICE", "INTERNETSERVICE", "ONLINESECURITY",
                                       "ONLINEBACKUP", "DEVICEPROTECTION", "TECHSUPPORT",
                                       "STREAMINGTV", "STREAMINGMOVIES"]]== "Yes").sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["STREAMINGTV"] == "Yes") or (x["STREAMINGMOVIES"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AUTO_PAYMENT"] = df["PAYMENTMETHOD"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_CHARGES"] = df["TOTALCHARGES"] / (df["TENURE"] + 1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_INCREASE"] = df["NEW_AVG_CHARGES"] / df["MONTHLYCHARGES"]

# Servis başına ücret
df["NEW_AVG_SERVICE_FEE"] = df["MONTHLYCHARGES"] / (df["NEW_TOTALSERVICE"] + 1)

df.info()
#Adım 3: Encoding işlemlerini gerçekleştiriniz.

cat_cols, num_cols, cat_but_car, num_but_cat= grab_col_names(df)
cat_cols
num_cols
cat_but_car
num_but_cat

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "float64"] #int yada float olmayan ve essiz iki sınıfa sahip olan degiskenleri sec
               and df[col].nunique() == 2]

for col in binary_cols:
    dff = label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in dff.columns if 10 >= dff[col].nunique() > 2]

dff= one_hot_encoder(dff, ohe_cols)

#Adım 4: Numerik değişkenler için standartlaştırma yapınız.

cat_cols, num_cols, cat_but_car, num_but_cat= grab_col_names(dff)
num_cols
"""ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols]) #standartlastırmayı barındıran nesneyi x degiskenine uygulayalım ve yeni sütun olarak ekleyelim ileride kıyaslama yapmak icin
df.head()"""

mms = MinMaxScaler()
dff["TENURE_MinMax_Scaler"] = mms.fit_transform(dff[["TENURE"]])
dff.describe().T

mms = MinMaxScaler()
dff["MONTHLYCHARGES_MinMax_Scaler"] = mms.fit_transform(dff[["MONTHLYCHARGES"]])

mms = MinMaxScaler()
dff["TOTALCHARGES_MinMax_Scaler"] = mms.fit_transform(dff[["TOTALCHARGES"]])

mms = MinMaxScaler()
dff["NEW_AVG_CHARGES_MinMax_Scaler"] = mms.fit_transform(dff[["NEW_AVG_CHARGES"]])

mms = MinMaxScaler()
dff["NEW_INCREASE_CHARGES_MinMax_Scaler"] = mms.fit_transform(dff[["NEW_INCREASE"]])

mms = MinMaxScaler()
dff["NEW_AVG_SERVICE_FEE_CHARGES_MinMax_Scaler"] = mms.fit_transform(dff[["NEW_AVG_SERVICE_FEE"]])


#Adım 5: Model oluşturunuz.

y = dff["CHURN"] #bagımlı degisken
X = dff.drop(["CUSTOMERID", "CHURN"], axis=1) #bagımsızlar bunların dısındakiler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
#veri setini ikiye ayırdım,train setiyle model kurucam,test setiyle kurdugum bu modeli test edicem

from sklearn.ensemble import RandomForestClassifier #model nesnesini getirelim,agac temelli yöntem

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train) #modeli kurduk
y_pred = rf_model.predict(X_test) #test setindeki x bagımsız degiskenlerini modele sorduk,y degiskenini tahmin et bakalım dedik
accuracy_score(y_pred, y_test) #test setinin y bagımlı degiskeniyle tahmin ettigim degeri kıyaslıyorum


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)


#accuracy,F1 skor  mülakatlarda cıkar ögren,TP TN  FP  FN  bunlara bak