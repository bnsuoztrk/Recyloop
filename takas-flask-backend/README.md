# Takas Flask Backend

Bu proje, sürdürülebilir takas platformu için geliştirilmiş Flask tabanlı bir backend uygulamasıdır.

## 🚀 Hızlı Başlangıç

### 1. Gereksinimler
- Python 3.8+
- pip

### 2. Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Uygulamayı başlat
python start.py
```

### 3. Çalıştırma

#### Ana Uygulama
```bash
python app.py
```

#### Test Sunucusu
```bash
python simple_server.py
```

## 📁 Proje Yapısı

```
takas-flask-backend/
├── app.py                 # Ana Flask uygulaması
├── models.py              # Veritabanı modelleri
├── blip_captioner.py      # BLIP görsel açıklama
├── efficientnet_detector.py # EfficientNet marka tespiti
├── simple_server.py       # Test sunucusu
├── start.py              # Başlatma scripti
├── requirements.txt      # Python bağımlılıkları
├── templates/
│   └── index.html        # Ana HTML template
├── static/
│   └── RecyLo.png        # Logo
├── uploads/              # Yüklenen dosyalar
├── instance/             # Veritabanı dosyaları
└── models/               # AI modelleri
```

## 🔧 Özellikler

### Backend API
- **Kullanıcı Yönetimi**: Kayıt, giriş, profil
- **Ürün Yönetimi**: Ürün ekleme, düzenleme, silme
- **Takas Sistemi**: Takas talepleri, durum takibi
- **AI Analiz**: EfficientNet marka tespiti, BLIP görsel açıklama
- **Kategori & Marka**: Önceden tanımlı kategoriler ve markalar

### Frontend
- **Responsive Tasarım**: Mobil uyumlu arayüz
- **Modern UI**: Tailwind CSS ile modern tasarım
- **Dosya Yükleme**: Drag & drop dosya yükleme
- **AI Entegrasyonu**: Gerçek zamanlı görsel analiz
- **Modal Sistemleri**: Giriş, kayıt, iş birliği formları

## 🌐 API Endpoints

### Kimlik Doğrulama
- `POST /api/auth/register` - Kullanıcı kaydı
- `POST /api/auth/login` - Giriş
- `GET /api/auth/me` - Kullanıcı bilgileri

### Ürünler
- `GET /api/items` - Ürün listesi
- `POST /api/items` - Ürün ekleme
- `GET /api/items/<id>` - Ürün detayı
- `PUT /api/items/<id>` - Ürün güncelleme
- `DELETE /api/items/<id>` - Ürün silme

### Takas
- `GET /api/trades` - Takas listesi
- `POST /api/trades` - Takas talebi
- `PUT /api/trades/<id>` - Takas güncelleme

### AI Analiz
- `POST /api/vlm/detect-brand` - EfficientNet marka tespiti
- `POST /api/blip/generate-caption` - BLIP görsel açıklama

## 🎯 Kullanım

1. **Ana Sayfa**: Platform tanıtımı ve iş birlikleri
2. **Nasıl Çalışır**: 4 adımlı süreç açıklaması
3. **Ürün Yükle**: AI destekli ürün analizi ve yayınlama
4. **Pazar Yeri**: Mevcut ürünleri görüntüleme ve takas talebi

## 🔧 Geliştirme

### Environment Variables
```bash
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=jwt-secret-string
DATABASE_URL=sqlite:///instance/takas.db
FLASK_ENV=development
```

### Veritabanı
SQLite veritabanı otomatik olarak oluşturulur. İlk çalıştırmada varsayılan kategoriler ve markalar eklenir.

### AI Modelleri
- **EfficientNet-B3**: Marka ve kategori tespiti
- **BLIP**: Görsel açıklama üretimi

## 🐛 Sorun Giderme

### Yaygın Sorunlar

1. **Model dosyası bulunamadı**
   - `models/` dizininde `efficientnetb3_prod_final.h5` 

2. **Bağımlılık hataları**
   - `pip install -r requirements.txt` 

3. **Port zaten kullanımda**
   - Farklı bir port kullanın: `python app.py --port 5001`

## 📝 Notlar

- Bu uygulama demo amaçlıdır
- Gerçek üretim için güvenlik önlemleri eklenmelidir
- AI modelleri CPU'da çalışır (GPU desteği için TensorFlow-GPU gerekir)


## 📄 Lisans


