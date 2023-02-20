# Telco-Customer-Churn
Dalam menghadapi dunia telekomunikasi yang serca cepat, diperlukan model machine learning untuk mendeteksi secara dini apakah customer akan churn atau tidak.

## Business understanding

**context**

Telco atau dalam terjemahaannya berarti perusahaan telepon, pada era yang serba cepat ini memiliki tingkat persaingan yang tinggi. masing pasing provider atau penyedia jasa jaringan mencoba berbagai macam cara untuk menghasilkan strategi yang tepat sasaran dan efektif. salah satu permasalahan yang dihadapi adalah mengenai customer churn atau pelanggan yang berhenti berlangganan dan beralih ke perusahaan kompetitor. dengan mengetahui kemungkinan customer churn, telco company dapat melakukan treatment khusus agar perilaku churn dapat dihindari.

untuk menghadapi kebutuhan yang cepat, seorang Data Scientist diminta untuk membuat model prediksi yang tepat untuk menentukan seorang pelanggan akan churn (beralih) atau tidak.

dalam prosesnya, diminta output berupa nilai 1 untuk customer churn dan 0 untuk tidak churn



**Problem Statement**

menurut [sumber](https://www.outboundengine.com/blog/customer-retention-marketing-vs-customer-acquisition-marketing/) , mencari customer baru memiliki biaya hingga 5 kali lebih besar dibandingkan dengan mempertahankan customer yang sudah ada. selanjutnya dengan meningkatkan nilai customer loyal sebanyak 5%, keuntungan dapat meningkat hingga 25-95%. Lebih spesifik pada [sumber berikut](https://www.revechat.com/blog/customer-acquisition-cost/) industri telco memerlukan $315 untuk mengakuisisi customer. 

Sehingga dalam mempertahankan pelanggan, sebuah perusahaan telco perlu memperlakukan customer yang ingin churn atau berpindah secara khusus. seperti memberikan potongan harga pada paket data tertentu, memberikan paket langgangan menarik, memberikan prioritas pelayanan lainnya dan upaya lainnya yang dapat meningkatkan loyalitas pelanggan. Namun untuk pemberikan perlakuan harus dilakukan secara selektif, perlakuan khusus harus dimaksimalkan untuk diberikan ke pelanggan yang memang akan berpindah sehingga tepat sasaran. 

**Approach**

model yang dapat digunakan adalah model klasifikasi untuk membantu perusahaan telco menentukan pelanggan yang akan churn atau tidak. dengan target:
* 0 : tidak berhenti berlangganan
* 1 : berhenti berlangganana


dan untuk menentukan nilai scoring, dilakukan analisis berikut
* Type 1 error : False Positive (pelanggan tidak churn tetapi diprediksi churn) Konsekuensi: insentif menjadi salah sasaran
* Type 2 error : False Negative (pelanggan churn tetapi diprediksi tidak akan churn) Konsekuensi: kehilangan pelanggan


Type 1 dan Type 2 error memiliki kesamaan kepentingan karena kesalahan berdsarkan fasle positive dan false negative akan memberikan kerugian pada peruasahaan. namun karena kebutuhan bisnis untuk mempertahankan customer memiliki kerugian lebih sedikit (dengan memberikan treatment), daripada kehilangan customer sama sekali. maka diterapkan nilai f2_score. sehingga tujuan utama pembuatan model yang mengutamakan mengurangi customer churn, namun tetap meminimalisir pemberian treatment kepada customer yang tidak churn.(nilai recall 2 kali lebih penting dari precision)


## DATA
Pada data_telco_customer_churn, terdapat features yang memiliki keterangan sebagai berikut
-	Dependents: Whether the customer has dependents or not.
-	Tenure: Number of months the customer has stayed with the company.
-	OnlineSecurity: Whether the customer has online security or not.
-	OnlineBackup: Whether the customer has online backup or not.
-	InternetService: Whether the client is subscribed to Internet service.
-	DeviceProtection: Whether the client has device protection or not.
-	TechSupport: Whether the client has tech support or not 
-	Contract: Type of contract according to duration.
-	PaperlessBilling: Bills issued in paperless form.
-	MonthlyCharges: Amount of charge for service on monthly bases.
-	Churn: Whether the customer churns or not.

## DATA PREPARATION

#### SCALING

pada model, akan dilakukan perbandingan model pada saat menggunakan scaling dan tidak menggunakan scaling, karena pada feature numerik tidak terdapat outlier, maka digunakan MinMaxScaler untuk melakukan scaling pada data

harapannya dengan menggunakan scaling, diperoleh hasil yang lebih baik dan skalatis (feature dengan variabel besar akan mendominasi fearture dnegan variabel kecil)

#### ENCODING

berdasarkan tabel diatas, encoding dilakukan menurut keterangan berikut : 
1. Dependents dengan menggunakan One Hot Encoder karena fitur ini tidak memiliki urutan ordinal dan jumlah uniqe data yang hanya 2
1. OnlineSecurity dengan menggunakan One Hot Encoder karena fitur ini tidak memiliki urutan ordinal dan jumlah uniqe data yang sedikit
1. OnlineBackup dengan menggunakan One Hot Encoder karena fitur ini tidak memiliki urutan ordinal dan jumlah uniqe data yang hanya sedikit
1. InternetService dengan menggunakan One Hot Encoder karena fitur ini tidak memiliki urutan ordinal dan jumlah uniqe data yang hanya sedikit
1. DeviceProtection dengan menggunakan One Hot Encoder karena fitur ini tidak memiliki urutan ordinal dan jumlah uniqe data yang hanya sedikit
1. TechSupport dengan menggunakan One Hot Encoder karena fitur ini tidak memiliki urutan ordinal dan jumlah uniqe data yang hanya sedikit
1. Contract dengan ordinal encoding karena nilainya bertingkat mulai dari month-to month, One Year dan Two Year
1. OnlineBackup dengan menggunakan One Hot Encoder karena fitur ini tidak memiliki urutan ordinal dan jumlah uniqe data yang hanya sedikit

#### ENCODING

data yang kita miliki memiliki imbalance data dan termasuk pada [mild imbalance](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data) dengan proporsi kelas minoritas sebesar 26.6%. agar distribusi kelas seimbang, kita akan menerapkan metode resampling yaitu Synthetic Minority Over-sampling Technique for Nominal and Continuous (SMOTENC), Hal ini dikarenakan :

* agar tidak menghilangkan informasi penting data pada kelas mayoritas, tidak digunakan undersammpling.
* untuk menghindari overfitting karena penduplikasian data yang telah ada sebelumnya sehingga pengklasifikasi terkena informasi yang sama jika menggunakan Random Oversampling. 
* pada data kita terdapat fitur yang numerikal (continuous) and kategorikal (nominal).

#### MODELING DAN EVALUATION
terlampir pada file

## CONCLUSION
* metric utama untuk scoring yang digunakan adalah f2_score karena recall dua kali lebih penting dibandingkan precision
* berdasarkan hyperparameter tunning dan komparasi antara interpretable dan explainable ML, model terbaik yaitu LightGBM memiliki parameter
    * scale_pos_weight : 3
    * max_depth : 2
    * learning_rate : 0.1
* berdasarkan visualisasi feature importance, fitur yang paling penting adalah tenure yang diikuti dengan MonthlyCharges dan Contract
* interpretasi SHAP : 
    * pelanggan dengan Contract yang lebih rendah, yaitu dengan kontrak month-to-month akan cenderung memiliki kemungkinan untuk churn dibandingkan dengan kontrak One Year atau Two Year 
    * pelanggan dengan tenure yang lebih pendek memiliki kecenderungan untuk churn dibandingkan dengan tenure yang lebih lama
    * pelanggan dengan monthly charge lebih tinggi akan memiliki kecenderungan untuk churn dibandingkan dengan monthly charge lebih rendah
    * pelanggan dengan intenet service fiber optik akan memiliki kecenderungan untuk churn dibandingkan dengan intenet service DSL atau No
    * pelanggan dengan internet dengan dependents atau tanggungan memiliki kecenderungan untuk churn

## RECOMENDATION

untuk menghadapi customer churn, dapat dilakukan beberapa langkah atau strategi berikut
* untuk pelanggan beralih contract menjadi jangka panjang, berikan insentif atau bonus lebih pada kontrak 1 tahun dan juga lebih banyak lagi pada 2 tahun yang bersifat jangka panjang
* Customer Loyalty Program, sehingga pelanggan akan tertarik untuk menjadi pelanggan tetap dengan adanya program tersebut. Loyalty program bisa berupa pemberian reward/hadiah sesuai dengan lamanya berlangganan sesuai dengan masa tenure.
* potongan harga ketika monthly charge sudah sampai pada titik tertentu sehingga adanya pembatasan monthly charge
* membuat paket tambahan layanan fiber optik dengan diskon
* paket khusus keluarga untuk menarik perhatian pihak yang memiliki tanggunan atau dependants dengan bonus yang signifikan sehingga membalikkan faktor dependant yang menjadi faktor untuk churn.

## LIMITATION

Model yang dihasilkan akan berlaku sesuai dengan batasan data yang ada, yaitu

* tenure antara 0 sampai dengan 72 bulan
* MonthlyCharges antara 18.8 sampai dengan 118.65
* Contract dalam jangka Month-to-month, One year, dan Two Year
* InternetService berupa 'DSL', 'Fiber Optic' dan 'No'
* Dependent, Paperless Billing dengan nilai 'Yes' atau 'No'
* OnlineSecurity, OnlineBackup, DeviceProtection, dan TechSupport berisi pilihan 'Yes', 'No' atau 'No internet service'.

sehingga selain dengan batasan ini, model menjadi tidak valid

## SAVING MODEL
best model lgbm di save melalui pickle
