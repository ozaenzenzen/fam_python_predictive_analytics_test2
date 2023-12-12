# Comparison KNN, Random Forest, Boosting Algorithm, and Support Vector Regression algorithm of Video Game Sales Dataset  - Fauzan Akmal Mahdi

## Domain Proyek
***
### Latar Belakang

Dalam ruang lingkup penjualan, setiap barang keluar dan masuk harus bisa terdokumentasi. Pendokumentasian atau pendataan barang keluar dan masuk harus dilakukan agar pihak penjual dapat melakukan analisis dari hasil penjualan yang telah dilakukan. Pendataan juga dapat membantu dalam merencanakan stok, promosi, dan pengembangan produk lebih lanjut. Dengan informasi ini, dapat dilakukan alokasi sumber daya secara lebih baik lagi. Dengan bantuan teknologi, lebih khususnya penggunaan *Machine Learning*, penarikan informasi dari sebuah data menjadi lebih efektif dan efisien. Pada penelitian ini ditujukan membandingkan kemampuan metode *machine learning* dalam memetakan penjualan global (*Global Sales*) dari data *Video Game Sales*. Selanjutnya, metode terbaik dari penelitian ini dapat digunakan untuk melakukan pemodelan terhadap data penjualan *video game* yang lain.

Referensi latar belakang dan metode 
- [*Data Visualization on video games global sales analysis & Predictive analysis on Real Estate pricing in Boston*](https://www.ijcstjournal.org/volume-9/issue-3/IJCST-V9I3P1.pdf)
- [*SALES PREDICTION USING MACHINE LEARNING ALGORITHMS*](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/files/13650914/IRJET_SALES_PREDICTION_USING_MACHINE_LEA.pdf)

## Business Understanding

### Problem Statements

Berdasarkan latar belakang yang telah diuraikan, terdapat beberapa *problem statements* yang akan dibahas sebagai berikut: 
- Bagaimana langkah dalam mempersiapkan data *Video Game Sales* untuk dilakukan pelatihan model *machine learning*?
- Bagaimana konfigurasi metode *machine learning* yang digunakan?
- Bagaimana kemampuan metode *machine learning* yang digunakan?

### Goals

Berdasarkan *Problem Statements* yang telah diuraikan, terdapat beberapa *Goals* yang akan dicapai sebagai berikut: 
- Mengetahui langkah dalam mempersiapkan data *Video Game Sales* untuk dilakukan pelatihan model *machine learning*?
- Mengetahui konfigurasi metode *machine learning* yang digunakan
- Mengetahui kemampuan metode *machine learning* yang digunakan

### Solution statements

Berdasarkan *Goals* yang telah diuraikan, terdapat beberapa rincian *Solution Statements* yang akan dicapai sebagai berikut: 
- Menggunakan teknik EDA (*Exploratory Data Analysis*) seperti Pendeskripsian Dataset, Penanganan *Missing Value*, *Univariate Analysis* dan *Multivariate Analysis*
- Menggunakan model machine learning *K-Nearest Neighbors* (KNN), *Random Forest*, *Boosting Algorithm*, dan *Support Vector Regression* untuk dapat dilakukan perbandingan dari setiap metode *machine learning* yang digunakan
- Membuat model *machine learning*
- Menggunakan metode evaluasi *Mean Squared Error* dan *R2 / R Squared Score*

## Data Understanding
Data yang digunakan bersumber dari [kaggle](https://www.kaggle.com/datasets/gregorut/videogamesales/data)

Spesifikasi dataset yang digunakan pada Table 1

Table 1. Informasi Dataset
| Jenis | Keterangan |
| --- | --- |
| Sumber Dataset | [Video Game Sales Dataset](https://www.kaggle.com/datasets/gregorut/videogamesales/data) |
| Kategori Dataset | Open Dataset |
| Lisensi Dataset | Unknown |
| Jenis Dataset | Comma-Separated Values  (CSV) |
| Ukuran Dataset |  1.36 MB |

Dataset merupakan koleksi data yang berfokus kepada penjualan video game. Dataset berisi berbagai fitur yang menggambarkan setiap game yang dijual, seperti  **'Rank', 'Name', 'Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'**.  Dataset berisi 11 kolom dan 16597 baris. Dengan 4 variabel bersifat kategorikal dan 7 variabel bersifat numerik. Tujuannya adalah untuk melakukan analisis prediksi Global Sales terhadap data penjualan video game. Berikut penjelasan lebih lengkap terhadap variabel dalam dataset

### Data Info:
#### Variabel-variabel pada Video Game Sales Dataset adalah sebagai berikut:
- Rank: Merupakan identifier angka dari baris dataset
- Name: Merupakan identifier nama game dari baris dataset
- Platform: Merupakan kolom yang berisi jenis platform dari game
- Year: Merupakan tahun dari perilisan game
- Genre: Merupakan kolom yang berisi jenis genre dari game
- Publisher: Merupakan kolom yang berisi jenis publisher dari game
- NA_Sales: Merupakan data penjualan di North America
- EU_Sales: Merupakan data penjualan di European
- JP_Sales: Merupakan data penjualan di Japan
- Other_Sales: Merupakan data penjualan diluar North America, European, dan Japan
- Global_Sales: Merupakan data penjualan Global

#### Deskripsi Dataset:
Didapatkan informasi dataset sebagai berikut: 
- 16598 baris, 11 kolom. 
- Tipe data: float64(6), int64(1), object(4)
- Size dataset: 1.4+ MB

Table 2. Informasi Dataset

<table><thead><tr><th>#</th><th>Column</th><th>Non-Null Count</th><th>Dtype</th></tr></thead><tbody><tr><td>---</td><td>------</td><td>--------------</td><td>-----</td></tr><tr><td>0</td><td>Rank</td><td>16598 non-null</td><td>int64</td></tr><tr><td>1</td><td>Name</td><td>16598 non-null</td><td>object</td></tr><tr><td>2</td><td>Platform</td><td>16598 non-null</td><td>object</td></tr><tr><td>3</td><td>Year</td><td>16327 non-null</td><td>float64</td></tr><tr><td>4</td><td>Genre</td><td>16598 non-null</td><td>object</td></tr><tr><td>5</td><td>Publisher</td><td>16540 non-null</td><td>object</td></tr><tr><td>6</td><td>NA_Sales</td><td>16598 non-null</td><td>float64</td></tr><tr><td>7</td><td>EU_Sales</td><td>16598 non-null</td><td>float64</td></tr><tr><td>8</td><td>JP_Sales</td><td>16598 non-null</td><td>float64</td></tr><tr><td>9</td><td>Other_Sales</td><td>16598 non-null</td><td>float64</td></tr><tr><td>10</td><td>Global_Sales</td><td>16598 non-null</td><td>float64</td></tr></tbody></table>

### *Check and Imputation Missing Value*:
#### *Check Missing Value*
Didapatkan *missing value* pada kolom `Year` dan `Publisher`. Kolom `Year` merupakan kolom numerik dengan 271 *missing value* dan kolom `Publisher` merupakan kolom kategorik dengan 58 *missing value*

Table 3. Informasi Missing Value

<table><thead><tr><th>Column</th><th>Count Null Value</th></tr></thead><tbody><tr><td>Rank</td><td>0</td></tr><tr><td>Name</td><td>0</td></tr><tr><td>Platform</td><td>0</td></tr><tr><td>Year</td><td>271</td></tr><tr><td>Genre</td><td>0</td></tr><tr><td>Publisher</td><td>58</td></tr><tr><td>NA_Sales</td><td>0</td></tr><tr><td>EU_Sales</td><td>0</td></tr><tr><td>JP_Sales</td><td>0</td></tr><tr><td>Other_Sales</td><td>0</td></tr><tr><td>Global_Sales</td><td>0</td></tr></tbody></table>

#### *Imputation Missing Value*
Hasil: Pada tahap ini digunakan
- teknik imputasi berdasarkan nilai rata-rata (*mean*) untuk kolom `Year`
- teknik imputasi berdasarkan nilai terbanyak (*most frequent*) untuk kolom `Publisher`
menggunakan library `from sklearn.impute import SimpleImputer`. Setelah proses imputasi, nilai null sudah tidak ditemukan

### *Check and Drop Outlier*:
Pada tahap ini dilakukan pengecekan terhadap outlier pada setiap kolom numerik yaitu **Year, NA_Sales, EU_Sales, JP_Sales, dan Other_Sales**. Kemudian dilakukan penghilangan baris yang memiliki outlier menggunakan metode *Inter Quartile Range* (IQR) sehingga didapatkan hasil 11926 baris dan 11 kolom

### *Univariate Analysis*:
*Univariate Analysis* adalah tahap untuk mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah untuk 1 jenis variabel / kolom. Tahap ini dilakukan dengan tujuan untuk mengetahui persebaran dan informasi nilai pada data per 1 kolom sehingga dapat dilakukan proses persiapan data yang sesuai.
#### Kolom Kategorik

- Kolom Genre

    Persebaran kolom Genre terdapat 12 nilai yang berbeda. Nilai tertinggi didapatkan pada value `Action` sebanyak 2409 dengan persentase 20.2% dan terendah pada value `Puzzle` sebanyak 448 dengan persentase 3.8%
    
    Image 1. Pesebaran Kolom Genre

    ![Kolom Genre](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/assets/67274784/11fd91ca-73a2-4054-b073-9765e736f296)

- Kolom Publisher

    Persebaran kolom Publisher terdapat 523 nilai yang berbeda. Nilai tertinggi didapatkan pada value `Electronic Arts` sebanyak 866 dengan persentase 7.3% dan nilai terendah pada beberapa value sebanyak 1

    Image 2. Pesebaran Kolom Publisher

    ![Kolom Publisher](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/assets/67274784/b06e4f72-6d68-47a9-b527-487f1063cf76)

- Kolom Platform

    Persebaran kolom Platform terdapat 30 nilai yang berbeda. Nilai tertinggi didapatkan pada value `DS` sebanyak 1754 dengan persentase 14.7% dan nilai terendah pada beberapa value yaitu `PCFX`, `NES`, `WS` dan `TG16` sebanyak 1
    
    Image 3. Pesebaran Kolom Platform

    ![Kolom Platform](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/assets/67274784/ed9bdf9c-f486-481e-8aaa-b458f6307ec2)


#### Kolom Numerik
Persebaran kolom numerik terdapat pada 7 kolom yaitu **'Rank', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'**. 
- Pada kolom `Rank` merupakan sebuah kolom identifier yang menunjukan jumlah baris pada dataset. 
- Pada kolom `Year` nilai terendah pada tahun 1980 dan untuk nilai tertinggi pada tahun 2020
- Pada kolom `NA_Sales`, `EU_Sales`, `JP_Sales`, `Other_Sales`, `Global_Sales` memilki rentang nilai antara 0 sampai 82.74 (Global_Sales)

Image 4. Pesebaran Kolom Numerik
![Kolom Numerik](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/assets/67274784/f23d31ad-91c6-4b91-83c5-9d85587ae7f7)

### *Multivariate Analysis*:
*Multivariate Analysis* yaitu tahap mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah untuk 2 atau lebih jenis variabel / kolom. Tahap ini dilakukan dengan tujuan untuk mengetahui persebaran dan informasi nilai pada data antara kolom dengan kolom lainnya sehingga dapat dilakukan proses persiapan data yang sesuai.

Pada tahap ini divisualisasikan persebaran kolom `EU_Sales` terhadap kolom kategorik `Genre`, `Publisher`, dan `Platform`

Image 5. Persebaran kolom `EU_Sales` terhadap kolom kategorik `Genre`, `Publisher`, dan `Platform`

![xx 5](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/assets/67274784/30b387c2-edb6-4288-8ecb-dcb08107e08c)
![download (1)](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/assets/67274784/ef135b47-164a-4dc7-b800-2c03efc1861b)
![download](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/assets/67274784/2c59895c-2ade-44e6-9d60-968aaef110c2)

Image 6. Visualisasi persebaran antar kolom numerik
![Visualisasi persebaran antar kolom numerik](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/assets/67274784/560dde9a-df97-432f-a0b0-c38ccd410962)


## Data Preparation
Langkah dalam persiapan data dijabarkan dalam beberapa poin
1. Mendownload dataset dari kaggle
    - Proses: Mengambil dataset dari website kaggle 
    - Alasan: Dataset akan digunakan untuk pengolahan dan pemodelan pada penelitian
2. Menyimpan dataset ke dalam Google Drive
    - Proses: Menyimpan dataset ke dalam penyimpanan cloud Google Drive
    - Alasan: Penyimpanan dataset dalam Google Drive agar mempermudah proses pemanggilan data dalam proses penelitian
3. Melakukan pemanggilan library pendukung
    - Proses: Melakukan pengimporan beberapa library dari numpy, matplotlib, pandas, dan seaborn dengan `import`
    - Alasan: Ditujukan untuk mendukung proses pengolahan, pelatihan, dan evaluasi data seperti pengolahan, visualisasi, dan proses matematis
4. Melakukan mounting Google Colab dengan Google Drive 
    - Proses: Melakukan koneksi antar platform Google Colab sebagai media pengolahan data dan Google Drive sebagai media penyimpanan data. Menggunakan library `from google.colab import drive` dan menggunakan kode `drive.mount('/content/drive')`
    - Alasan: Agar proses persiapan data dan pengolahan model bisa dilakukan dari data yang disimpan dalam Google Drive dan media pengolahan data di Google Colab
5. Memanggil dataset 
    - Proses: Melakukan pemanggilan dataset dengan library pandas `pd.read_csv(url)`
    - Alasan: Agar dataset siap digunakan pada proses pengolahan dalam Google Colab
6. Melakukan pengecekan informasi dataset
    - Proses: Pengecekan informasi dataset terbagi menjadi sub tahap seperti
        - menggunakan `dataset.info()` untuk menampilkan jumlah baris, missing value, dan tipe data dari dataset
        - menggunakan `dataset.describe()` untuk mendeskripsikan parameter singkat pada dataset
            - Count atau jumlah baris dari setiap kolom
            - Mean atau rata-rata dari setiap kolom
            - Standar deviasi dari setiap kolom
            - Nilai minimum / terkecil dari setiap kolom
            - Nilai kuartil pertama atau 25% dari setiap kolom
            - Nilai kuartil kedua atau 50% atau median dari setiap kolom
            - Nilai kuartil ketiga atau 75% dari setiap kolom
            - Nilai maximum / terbesar dari setiap kolom
        - menggunakan `dataset.column` untuk mengetahui kolom yang tersedia pada dataset
    - Alasan: Untuk mengetahui informasi dasar dari dataset yang akan digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data
7. Melakukan pengecekan missing value dalam data (bisa NULL atau 0) 
    - Proses: Menggunakan `dataset.isnull().sum()` untuk mendapatkan informasi apakah terdapat data yang bersifat null
    - Alasan: Untuk mengetahui informasi missing value dari dataset yang akan digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data
8. Melakukan imputasi missing value 
    - Proses: Menggunakan `from sklearn.impute import SimpleImputer` menggunakan metode khusus dalam melakukan imputasi terhadap kolom yang terdapat missing value. 
        - Imputasi menggunakan nilai rata-rata (mean) untuk kolom `Year`
        - Imputasi menggunakan nilai terbanyak (most_frequent) untuk kolom `Publisher`
    - Alasan: Agar proses pengolahan lancar karena missing value dalam dataset sudah diimputasi / diisi dengan metode khusus yang dipilih
9. Melakukan outlier analisis & memvisualisasikan persebaran data pada setiap kolom untuk mengetahui outlier = sebuah data atau observasi yang menyimpang secara ekstrim dari rata-rata sekumpulan data yang ada
    - Proses: Menggunakan fungsi `boxplot()` dari seaborn untuk mendapatkan informasi visualisasi persebaran data dari setiap kolom
    - Alasan: Untuk mengetahui informasi persebaran nilai apakah terdapat nilai outlier dari dataset yang akan digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data
10. Drop Outlier yaitu menangani outlier pada dataset. Disini digunakan metode Inter Quartile Range (IQR)
    - Proses: Menggunakan metode Inter Quartile Range (IQR) untuk menghilangkan nilai outlier pada dataset sehingga didapatkan informasi dataset 11926 baris dan 11 kolom
    - Alasan: Proses penghilangan outlier dilakukan karena terdapat nilai outlier pada kolom **Year** 
11. Melakukan univariate analysis yaitu mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah untuk 1 jenis variabel / kolom
    - Proses: Proses dalam tahap ini ditujukan untuk mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah untuk 1 jenis variabel / kolom. Tahap ini terbagi menjadi sub tahap seperti berikut
        - Membagi kolom yang bersifat numerk dan kategorikal 
        - Menyimpan kolom ke dalam masing-masing variabel yaitu `numerical_features` dan `categorical_features`
        - Melakukan visualisasi untuk menginformasikan data pada kolom kategorik yaitu `Genre`, `Publisher` dan `Platform`
            - Didapatkan hasil pada kolom `Genre` pada Table 2
            - Table 4. hasil univariate analysis pada kolom kategorik `country`
            <table><thead><tr><th>Data</th><th>jumlah sampel</th><th>persentase</th></tr></thead><tbody><tr><td>Action</td><td>2409</td><td>20.2</td></tr><tr><td>Sports</td><td>1620</td><td>13.6</td></tr><tr><td>Misc</td><td>1336</td><td>11.2</td></tr><tr><td>Adventure</td><td>1120</td><td>9.4</td></tr><tr><td>Racing</td><td>936</td><td>7.8</td></tr><tr><td>Shooter</td><td>909</td><td>7.6</td></tr><tr><td>Role-Playing</td><td>863</td><td>7.2</td></tr><tr><td>Simulation</td><td>648</td><td>5.4</td></tr><tr><td>Platform</td><td>591</td><td>5.0</td></tr><tr><td>Fighting</td><td>541</td><td>4.5</td></tr><tr><td>Strategy</td><td>505</td><td>4.2</td></tr><tr><td>Puzzle</td><td>448</td><td>3.8</td></tr></tbody></table>
            - Didapatkan hasil pada kolom `Platform` pada Table 3
            - Table 5. hasil univariate analysis pada kolom kategorik `Platform`
            <table><thead><tr><th>Data</th><th>jumlah sampel</th><th>persentase</th></tr></thead><tbody><tr><td>DS</td><td>1754</td><td>14.7</td></tr><tr><td>PS2</td><td>1466</td><td>12.3</td></tr><tr><td>Wii</td><td>1035</td><td>8.7</td></tr><tr><td>X360</td><td>928</td><td>7.8</td></tr><tr><td>PSP</td><td>906</td><td>7.6</td></tr><tr><td>PC</td><td>821</td><td>6.9</td></tr><tr><td>PS3</td><td>814</td><td>6.8</td></tr><tr><td>XB</td><td>735</td><td>6.2</td></tr><tr><td>PS</td><td>730</td><td>6.1</td></tr><tr><td>GBA</td><td>681</td><td>5.7</td></tr><tr><td>GC</td><td>479</td><td>4.0</td></tr><tr><td>3DS</td><td>344</td><td>2.9</td></tr><tr><td>PSV</td><td>343</td><td>2.9</td></tr><tr><td>PS4</td><td>222</td><td>1.9</td></tr><tr><td>N64</td><td>221</td><td>1.9</td></tr><tr><td>XOne</td><td>160</td><td>1.3</td></tr><tr><td>WiiU</td><td>108</td><td>0.9</td></tr><tr><td>SAT</td><td>69</td><td>0.6</td></tr><tr><td>SNES</td><td>44</td><td>0.4</td></tr><tr><td>DC</td><td>22</td><td>0.2</td></tr><tr><td>2600</td><td>13</td><td>0.1</td></tr><tr><td>GEN</td><td>10</td><td>0.1</td></tr><tr><td>NG</td><td>6</td><td>0.1</td></tr><tr><td>SCD</td><td>4</td><td>0.0</td></tr><tr><td>GB</td><td>4</td><td>0.0</td></tr><tr><td>3DO</td><td>3</td><td>0.0</td></tr><tr><td>PCFX</td><td>1</td><td>0.0</td></tr><tr><td>NES</td><td>1</td><td>0.0</td></tr><tr><td>WS</td><td>1</td><td>0.0</td></tr><tr><td>TG16</td><td>1</td><td>0.0</td></tr></tbody></table>
            - Didapatkan hasil pada kolom `Publisher` pada Table 4
            - Table 6. hasil univariate analysis pada kolom kategorik `Publisher`. Ditampilkan 5 teratas dan 5 terbawah karena terdapat 523 kategori
            <table><thead><tr><th>Data</th><th>jumlah sampel</th><th>persentase</th></tr></thead><tbody><tr><td>Electronic Arts</td><td>866</td><td>7.3</td></tr><tr><td>Activision</td><td>734</td><td>6.2</td></tr><tr><td>Ubisoft</td><td>719</td><td>6.0</td></tr><tr><td>THQ</td><td>568</td><td>4.8</td></tr><tr><td>Namco Bandai Games</td><td>566</td><td>4.7</td></tr><tr><td>...</td><td>...</td><td>...</td></tr><tr><td>Marvel Entertainment</td><td>1</td><td>0.0</td></tr><tr><td>Illusion Softworks</td><td>1</td><td>0.0</td></tr><tr><td>Phantagram</td><td>1</td><td>0.0</td></tr><tr><td>The Learning Company</td><td>1</td><td>0.0</td></tr><tr><td>UIG Entertainment</td><td>1</td><td>0.0</td></tr><tr><td></td><td></td><td></td></tr><tr><td>[523 rows x 2 columns]</td><td></td><td></td></tr></tbody></table>
        - Melakukan visualisasi persebaran nilai dalam bentuk grafik untuk menginformasikan data pada kolom numerik yaitu **'Rank', 'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'**
    - Alasan: Untuk mengetahui informasi persebaran nilai pada kolom kategorikal dan numerikal secara spesifik univariate per kolom pada dataset yang digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data
12. Melakukan multivariate analysis yaitu mengeksplorasi dan menjelaskan setiap variabel dalam kumpulan data secara terpisah untuk 2 atau lebih jenis variabel / kolom
    - Proses: Pada tahap ini terbagi menjadi beberapa sub tahap sebagai berikut
        - Visualisasi data kategorik `Genre`, `Publisher` dan `Platform` terhadap data numerik yang dipilih `Year`, `NA_Sales`, `EU_Sales`, `JP_Sales`, dan `Other_Sales` menggunakan fungsi `pairplot()`
        - Visualisasi antar data kolom numerik menggunakan fungsi `pairplot()`
        - Visualisasi matriks korelasi antar data kolom numerik untuk mendapatkan nilai koefisien korelasi
    - Alasan: Untuk mengetahui informasi persebaran nilai antar kolom kategorikal dan numerikal dan kolom numerik dengan kolom numerik lainnya  pada dataset yang digunakan sehingga dapat ditentukan langkah selanjutnya dalam melakukan persiapan data. 
13. Melakukan penghilangan kolom yang tidak diperlukan sesuai analisis masalah dan tujuan penelitian
    - Proses: Menghilangkan kolom fitur yang tidak diperlukan yaitu `Rank` menggunakan fungsi `drop()`
    - Alasan: Korelasi kolom `Rank` terhadap dataset sangat kecil karena secara sifatnya yang hanya sebuah identifier unik pada baris kolom
14. Melakukan Encoding Categorical Features yaitu memberikan alias dalam bentuk numerik kepada kolom yang bersifat kategorikal
    - Proses: Melakukan pemberian alias terhadap kolom kategorik agar bisa dapat berbentuk numerik dan memisahkan data kategorik sebagai kolom terpisah dengan subtahap sebagai berikut
        - Melakukan encode terhadap kolom kategorik
        - Memisahkan data kategorik sebagai kolom terpisah
        - Menggabungkan kolom hasil encode ke dalam dataset utama
        - Menghapus kolom kategorik yang lama
    - Alasan: Pemberian alias terhadap kolom kategorik `Genre`, `Publisher` dan `Platform` ditujukan agar kolom fitur tersebut dapat digunakan dalam proses pemodelan
15. Melakukan pembagian dataset menjadi data train dan data test dalam pembagian yang ditentukan
    - Proses: Melakukan pembagian data test dan data train dengan pembagian 10% data test dan 90% data train menggunakan fungsi `train_test_split(X, y, test_size = 0.1, random_state = 123)`
    - Alasan: Pembagian dataset disesuaikan agar tidak terjadi underfit atau overfit pada proses pemodelan yang akan dilakukan dengan metode machine learning
16. Melakukan standarisasi atau perubahan skala nilai pada suatu kolom sesuai skala yang diinginkan. Fitur kolom yang menjadi tujuan adalah kolom `Global_Sales`
    - Proses: Melakukan standarisasi nilai pada kolom `'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'` dengan menggunakan funsi dari sklearn yaitu `StandardScaler`
    - Alasan: Proses standarisasi data dilakukan agar rentang nilai pada kolom `'Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'` tidak terlampau jauh dan agar menyelaraskan dengan kolom dataset lain

## Modeling
Algoritma Penelitian ini melakukan pemodelan dengan 4 algoritma, yaitu K-Nearest Neighbour, Random Forest, dan Support Vector Regression

- **K-Nearest Neighbor**: K-Nearest Neighbor merupakan salah satu algoritma supervised machine learning yang memiliki cara kerja membandingkan jarak satu sampel ke sampel pelatihan/training lain dengan menentukan sejumlah k tetangga terdekat [3]. Karena proyek ini bertujuan untuk proses regresi, maka digunakan metode KNN untuk regresi yaitu K Neighbors Regressor. Proyek ini menggunakan library `sklearn.neighbors.KNeighborsRegressor` dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :

    - `n_neighbors = Jumlah k tetangga tedekat.`

- **Random Forest**: Random forest adalah salah satu metode dalam machine learning dengan metode ensemble. Cara kerja metode ini adalah dengan membangun banyak decision tree pada waktu pelatihan/training [3]. Karena proyek ini bertujuan untuk proses regresi, maka digunakan metode Random Forest untuk regresi yaitu Random Forest Regressor. Proyek ini menggunakan `sklearn.ensemble.RandomForestRegressor` dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :

    - `n_estimators = Jumlah maksimum estimator di mana boosting dihentikan.`
    - `max_depth = Kedalaman maksimum setiap tree.`
    - `random_state = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.`

- **Adaboost**: AdaBoost atau Adaptive Boosting adalah metode dalam machine learning dengan metode ensemble. Algoritma yang paling umum digunakan dengan AdaBoost adalah decision tree satu tingkat yang berarti memiliki pohon Keputusan dengan hanya 1 split.[4] Algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi dengan cara menggabungkan beberapa model sederhana dan dianggap lemah secara berurutan sehingga membentuk suatu model yang kuat.[5] Proyek ini menggunakan `sklearn.ensemble.AdaBoostRegressor` dengan memasukkan X_train dan y_train dalam membangun model. Parameter yang digunakan pada proyek ini adalah :

    - `n_estimators = Jumlah maksimum estimator di mana boosting dihentikan.`
    - `learning_rate = Learning rate memperkuat kontribusi setiap regressor.`
    - `random_state = Mengontrol seed acak yang diberikan pada setiap base_estimator pada setiap iterasi boosting.`
    
- **Support Vector Machine**: Support Vector Machine (SVM) adalah metode supervised machine learning yang digunakan untuk tugas klasifikasi dan regresi. Tujuan utama SVM adalah untuk menemukan optimal hyperplane yang memisahkan titik data milik kelas berbeda dalam ruang fitur [6]. Ini sangat efektif dalam ruang berdimensi tinggi dan banyak digunakan untuk tugas-tugas seperti klasifikasi gambar, klasifikasi teks, dan banyak lagi. Karena proyek ini bertujuan untuk proses regresi, maka digunakan metode SVM untuk regresi yaitu Support Vector Regression. Proyek ini menggunakan `from sklearn.svm import SVR` dengan memasukkan X_train dan y_train dalam membangun model.


## Evaluation
### Metrik Penilaian Evaluasi
Metrik evaluasi yang digunakan pada proyek ini adalah *mean squared error* (MSE) dan *R square* (*coefficient of determination*). 
- **Akurasi** menentukan tingkat kemiripan antara hasil prediksi dengan nilai yang sebenarnya (y_test). 
- ***Mean squared error* (MSE)** mengukur error dalam model statistik dengan cara menghitung rata-rata error dari kuadrat hasil aktual dikurang hasil prediksi. 
- ***R Squared* (R2)** merupakan suatu nilai yang memperlihatkan seberapa besar variabel independen (eksogen) mempengaruhi variabel dependen (endogen). R squared merupakan angka yang berkisar antara 0 sampai 1 yang mengindikasikan besarnya kombinasi variabel independen secara bersama – sama mempengaruhi nilai variabel dependen. Nilai R-squared (R2) digunakan untuk menilai seberapa besar pengaruh variabel laten independen tertentu terhadap variabel laten dependen.[1] Berikut formula MSE :

### Berikut hasil evaluasi
- Akurasi

    Table 7. Hasil Akurasi Setiap Model

    <table><thead><tr><th align="right"></th><th align="right">KNN</th><th align="right">RF</th><th align="right">Adaboost</th><th align="right">SVR</th></tr></thead><tbody><tr><td align="right">accuracy</td><td align="right">0.972663</td><td align="right">0.996812</td><td align="right">0.918662</td><td align="right">0.891426</td></tr></tbody></table>

- Mean Squred Error (MSE)

    Image 7. Hasil *Mean Squared Error* (MSE) pada setiap model untuk *data train* dan *data test*

    ![*dicoding mahcine learning*](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/assets/67274784/7395d2c3-7101-4394-a590-6577990be6ab)

- R square

    Table 8. Hasil *R Squared* Setiap Model

    <table><thead><tr><th align="right"></th><th align="right">KNN</th><th align="right">RF</th><th align="right">Adaboost</th><th align="right">SVR</th></tr></thead><tbody><tr><td align="right">r2_score</td><td align="right">0.941931</td><td align="right">0.967273</td><td align="right">0.866186</td><td align="right">0.891318</td></tr></tbody></table>

### Kesimpulan

Pada evaluasi dengan parameter Akurasi dan *R Squared*, semakin tinggi maka semakin bagus model yang sudah dilatih. Hasil evaluasi dengan nilai Akurasi dan *R Squared* tertinggi pada metode *Random Forest* yaitu akurasi sebesar 0.996812 (atau 99,68%) dan *R Squared Score* sebesar 0.967273 (atau 96,72%). Sedangkan nilai Akurasi dan *R Squared* paling rendah pada metode *Support Vector Regression* yaitu akurasi 0.891426 (atau 89,14%) dan *R Squared Score* 0.891318 (atau 89,13%).

Dari 4 model algoritma yang digunakan, dapat disimpulkan hasil perbandingan serta visualisasi perbandingan prediksi 4 algoritma yaitu KNN, *Random Forest*, *AdaBoost*, dan *Support Vector Regression* sebagai berikut. Model *Random Forest* memiliki nilai error pada ***data test*** maupun ***data train*** yang paling kecil kemudian model dengan KNN memiliki nilai error yang sedikit lebih banyak daripada model *Random Forest* sedangkan model *AdaBoost* miliki lebih banyak nilai error dibandingkan *Random Forest* dan KNN namum model *Support Vector Regression* memiliki nilai error paling banyak dari ketiga model *Random Forest*, KNN, dan *AdaBoost*. Secara keseluruhan hasil MSE pada ***data train*** lebih kecil dibandingkan ***data test*** pada setiap model yang diuji.


## Referensi
[1] [MEMAHAMI R SQUARE (KOEFISIEN DETERMINASI) DALAM PENELITIAN ILMIAH](https://accounting.binus.ac.id/2021/08/12/memahami-r-square-koefisien-determinasi-dalam-penelitian-ilmiah/) 

[2] [Akhil, T.S., Kaur, N., Reddy, B.S., Vinay, B. and Kishore, K.N., Data Visualization on video games global sales analysis& Predictive analysis on Real Estate pricing in Boston.](https://www.ijcstjournal.org/volume-9/issue-3/IJCST-V9I3P1.pdf)

[3] [Bajaj, P., Ray, R., Shedge, S., Vidhate, S. and Shardoor, N., 2020. Sales prediction using machine learning algorithms. International Research Journal of Engineering and Technology (IRJET), 7(6), pp.3619-3625.
](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/files/13650914/IRJET_SALES_PREDICTION_USING_MACHINE_LEA.pdf)

[4] [Xiao, F., Wang, Y., He, L., Wang, H., Li, W. and Liu, Z., 2019. Motion estimation from surface electromyogram using adaboost regression and average feature values. IEEE Access, 7, pp.13121-13134.
](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/files/13651059/xiao2019.pdf)

[5] [Zhu, X., Zhang, P. and Xie, M., 2021. A Joint Long Short-Term Memory and AdaBoost regression approach with application to remaining useful life estimation. Measurement, 170, p.108707.](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/files/13651085/zhu2020.pdf)

[6] [Zhang, F. and O'Donnell, L.J., 2020. Support vector regression. In Machine learning (pp. 123-140). Academic Press.
](https://github.com/ozaenzenzen/fam_python_predictive_analytics_test2/files/13651086/10.1016%40B978-0-12-815739-8.00007-9.pdf)
