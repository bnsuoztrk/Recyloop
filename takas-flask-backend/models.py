from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

db = SQLAlchemy()

class User(db.Model):
    """Kullanıcı modeli"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.Text, nullable=True)
    profile_image = db.Column(db.String(200), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    is_verified = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # İlişkiler
    items = db.relationship('Item', backref='owner', lazy=True, cascade='all, delete-orphan')
    sent_trades = db.relationship('Trade', foreign_keys='Trade.sender_id', backref='sender', lazy=True)
    received_trades = db.relationship('Trade', foreign_keys='Trade.receiver_id', backref='receiver', lazy=True)
    
    def set_password(self, password):
        """Şifreyi hash'le"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Şifreyi kontrol et"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Kullanıcı bilgilerini dictionary olarak döndür"""
        return {
            'id': self.id,
            'uuid': self.uuid,
            'name': self.name,
                'email': self.email,
                'phone': self.phone,
            'address': self.address,
            'profile_image': self.profile_image,
                'is_active': self.is_active,
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Category(db.Model):
    """Kategori modeli"""
    __tablename__ = 'categories'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)
    icon = db.Column(db.String(50), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # İlişkiler
    items = db.relationship('Item', backref='category', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon': self.icon,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Item(db.Model):
    """Ürün modeli"""
    __tablename__ = 'items'
    
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    brand = db.Column(db.String(100), nullable=True)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'), nullable=False)
    condition = db.Column(db.String(50), nullable=False)  # Mükemmel, İyi, Orta, Kullanılmış, Hasarlı
    estimated_value = db.Column(db.String(50), nullable=False)  # 0-50₺, 50-100₺, etc.
    image_path = db.Column(db.String(500), nullable=False)
    ai_analysis_data = db.Column(db.JSON, nullable=True)  # AI analiz sonuçları
    is_available = db.Column(db.Boolean, default=True)
    is_featured = db.Column(db.Boolean, default=False)
    view_count = db.Column(db.Integer, default=0)
    owner_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # İlişkiler
    trades = db.relationship('Trade', backref='item', lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'uuid': self.uuid,
            'title': self.title,
            'description': self.description,
            'brand': self.brand,
            'category': self.category.name if self.category else None,
            'condition': self.condition,
            'estimated_value': self.estimated_value,
            'image_path': self.image_path,
            'ai_analysis_data': self.ai_analysis_data,
            'is_available': self.is_available,
            'is_featured': self.is_featured,
            'view_count': self.view_count,
            'owner': {
                'id': self.owner.id,
                'name': self.owner.name,
                'email': self.owner.email
            } if self.owner else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Trade(db.Model):
    """Takas modeli"""
    __tablename__ = 'trades'
    
    id = db.Column(db.Integer, primary_key=True)
    uuid = db.Column(db.String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    item_id = db.Column(db.Integer, db.ForeignKey('items.id'), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, accepted, rejected, completed, cancelled
    message = db.Column(db.Text, nullable=True)
    counter_offer = db.Column(db.Text, nullable=True)  # Karşı teklif
    sender_item_info = db.Column(db.JSON, nullable=True)  # Gönderenin ürün bilgileri
    meeting_location = db.Column(db.String(200), nullable=True)
    meeting_date = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'uuid': self.uuid,
            'item': self.item.to_dict() if self.item else None,
            'sender': {
                'id': self.sender.id,
                'name': self.sender.name,
                'email': self.sender.email
            } if self.sender else None,
            'receiver': {
                'id': self.receiver.id,
                'name': self.receiver.name,
                'email': self.receiver.email
            } if self.receiver else None,
            'status': self.status,
            'message': self.message,
            'counter_offer': self.counter_offer,
            'sender_item_info': self.sender_item_info,
            'meeting_location': self.meeting_location,
            'meeting_date': self.meeting_date.isoformat() if self.meeting_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Brand(db.Model):
    """Marka modeli"""
    __tablename__ = 'brands'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    category = db.Column(db.String(50), nullable=True)  # electronics, fashion, automotive, etc.
    logo_url = db.Column(db.String(500), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'logo_url': self.logo_url,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class AIAnalysis(db.Model):
    """AI analiz geçmişi"""
    __tablename__ = 'ai_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('items.id'), nullable=True)
    image_path = db.Column(db.String(500), nullable=False)
    brand_detected = db.Column(db.String(100), nullable=True)
    confidence_score = db.Column(db.Float, nullable=True)
    model_used = db.Column(db.String(100), nullable=True)
    analysis_data = db.Column(db.JSON, nullable=True)
    processing_time = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'item_id': self.item_id,
            'image_path': self.image_path,
            'brand_detected': self.brand_detected,
            'confidence_score': self.confidence_score,
            'model_used': self.model_used,
            'analysis_data': self.analysis_data,
            'processing_time': self.processing_time,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Veritabanı başlatma fonksiyonu
def init_db(app):
    """Veritabanını başlat"""
    db.init_app(app)
    
    with app.app_context():
        # Tabloları oluştur
        db.create_all()
        
        # Varsayılan kategorileri ekle
        create_default_categories()

def create_default_categories():
    """AI model kategorileri ile uyumlu varsayılan kategorileri oluştur"""
    categories = [
    # Shoes (Ayakkabı)
    {'name': 'babet', 'description': 'Babet ayakkabı', 'icon': 'fas fa-shoe-prints'},
    {'name': 'cizme', 'description': 'Çizme ayakkabı', 'icon': 'fas fa-shoe-prints'},
    {'name': 'bot', 'description': 'Bot ayakkabı', 'icon': 'fas fa-shoe-prints'},
    {'name': 'stiletto', 'description': 'Stiletto ayakkabı', 'icon': 'fas fa-shoe-prints'},
    {'name': 'spor-ayakkabi', 'description': 'Spor ayakkabı', 'icon': 'fas fa-shoe-prints'},
    {'name': 'sandalet', 'description': 'Sandalet ayakkabı', 'icon': 'fas fa-shoe-prints'},
    {'name': 'terlik', 'description': 'Terlik ayakkabı', 'icon': 'fas fa-shoe-prints'},

    # Bags (Çanta)
    {'name': 'cuzdan', 'description': 'Cüzdan', 'icon': 'fas fa-wallet'},
    {'name': 'sirt-cantasi', 'description': 'Sırt çantası', 'icon': 'fas fa-backpack'},
    {'name': 'laptop-cantasi', 'description': 'Laptop çantası', 'icon': 'fas fa-laptop'},
    {'name': 'kol-cantasi', 'description': 'Kol çantası', 'icon': 'fas fa-handbag'},

    # Clothing (Giyim)
    {'name': 'tshirt', 'description': 'T-shirt', 'icon': 'fas fa-tshirt'},
    {'name': 'sweatshirt', 'description': 'Sweatshirt', 'icon': 'fas fa-tshirt'},
    {'name': 'sort', 'description': 'Şort', 'icon': 'fas fa-user'},
    {'name': 'pantolon', 'description': 'Pantolon', 'icon': 'fas fa-user'},
    {'name': 'mont', 'description': 'Mont', 'icon': 'fas fa-vest'},
    {'name': 'kot-ceket', 'description': 'Kot ceket', 'icon': 'fas fa-vest'},
    {'name': 'kazak', 'description': 'Kazak', 'icon': 'fas fa-tshirt'},
    {'name': 'kaban', 'description': 'Kaban', 'icon': 'fas fa-vest'},
    {'name': 'gomlek', 'description': 'Gömlek', 'icon': 'fas fa-tshirt'},
    {'name': 'etek', 'description': 'Etek', 'icon': 'fas fa-female'},
    {'name': 'elbise', 'description': 'Elbise', 'icon': 'fas fa-female'},
    {'name': 'deri-ceket', 'description': 'Deri ceket', 'icon': 'fas fa-vest'},
    {'name': 'bluz', 'description': 'Bluz', 'icon': 'fas fa-tshirt'}
]

    
    for cat_data in categories:
        existing = Category.query.filter_by(name=cat_data['name']).first()
        if not existing:
            category = Category(**cat_data)
            db.session.add(category)
    
    db.session.commit()

