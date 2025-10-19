from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import os
from datetime import datetime, timedelta
import uuid
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from efficientnet_detector import efficientnet_detector
from blip_captioner import generate_image_caption
from models import db, User, Item, Trade, Category, Brand, AIAnalysis, init_db

# Environment variables
load_dotenv()

# === 1️⃣ Özel focal loss fonksiyonunu tanımla ===
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_crossentropy

# === 2️⃣ Classes dizisi (AI model kategorileri) ===
CLASSES = [
    "babet", "bluz", "bot", "cizme", "cuzdan", "deri-ceket", "elbise", "etek",
    "gomlek", "kaban", "kazak", "kol-cantasi", "kot-ceket", "laptop-cantasi",
    "mont", "pantolon", "sandalet", "sirt-cantasi", "sort", "spor-ayakkabi",
    "stiletto", "sweatshirt", "terlik", "tshirt"
]

# === 3️⃣ EfficientNet preprocessing fonksiyonu (doğru versiyon) ===
def preprocess_image_efficientnet(image_path, target_size=(300, 300)):
    """EfficientNet için doğru görsel ön işleme"""
    try:
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.efficientnet import preprocess_input
        
        # Keras image loading kullan
        img = image.load_img(image_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # EfficientNet için özel preprocessing
        return img_array
    except Exception as e:
        print(f"Preprocessing hatası: {e}")
        return None

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///takas.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret-string')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(days=7)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
CORS(app)
jwt = JWTManager(app)

# Initialize database
init_db(app)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api')
def api_info():
    return jsonify({
        'message': 'Takas API - Sustainable Item Exchange Platform',
        'version': '1.0.0',
        'endpoints': {
            'auth': '/api/auth',
            'items': '/api/items',
            'trades': '/api/trades',
            'users': '/api/users',
            'ai': '/api/ai'
        }
    })

# Authentication routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data.get('name') or not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Name, email and password are required'}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=data['email']).first():
            return jsonify({'error': 'User already exists'}), 400
        
        # Create new user
        user = User(
            name=data['name'],
            email=data['email'],
            phone=data.get('phone'),
            address=data.get('address')
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'User created successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        
        if not data.get('email') or not data.get('password'):
            return jsonify({'error': 'Email and password are required'}), 400
        
        user = User.query.filter_by(email=data['email']).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Create access token
        access_token = create_access_token(identity=user.id)
        
        return jsonify({
            'access_token': access_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    try:
        user_id = get_jwt_identity()
        user = User.query.get(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'user': user.to_dict()}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Items routes
@app.route('/api/items', methods=['GET'])
def get_items():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        category = request.args.get('category')
        search = request.args.get('search')
        min_price = request.args.get('min_price', type=float)
        max_price = request.args.get('max_price', type=float)
        
        query = Item.query.filter_by(is_available=True)
        
        if category:
            query = query.join(Category).filter(Category.name == category)
        
        if search:
            query = query.filter(
                Item.title.contains(search) | 
                Item.description.contains(search) |
                Item.brand.contains(search)
            )
        
        if min_price is not None:
            query = query.filter(Item.price >= min_price)
        
        if max_price is not None:
            query = query.filter(Item.price <= max_price)
        
        items = query.order_by(Item.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        
        return jsonify({
            'items': [item.to_dict() for item in items.items],
            'total': items.total,
            'pages': items.pages,
            'current_page': page
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/items', methods=['POST'])
@jwt_required()
def create_item():
    try:
        user_id = get_jwt_identity()
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get form data
        title = request.form.get('title', 'Untitled Item')
        description = request.form.get('description', '')
        brand = request.form.get('brand', '')
        category_name = request.form.get('category', 'Diğer')
        condition = request.form.get('condition', 'Orta')
        estimated_value = request.form.get('price', '0-50₺')
        
        # Find category
        category = Category.query.filter_by(name=category_name).first()
        if not category:
            category = Category.query.filter_by(name='Diğer').first()
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Create new item
        item = Item(
            title=title,
            description=description,
            brand=brand,
            category_id=category.id,
            condition=condition,
            estimated_value=estimated_value,
            image_path=file_path,
            owner_id=user_id
        )
        
        db.session.add(item)
        db.session.commit()
        
        return jsonify({
            'message': 'Item created successfully',
            'item': item.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    try:
        item = Item.query.get_or_404(item_id)
        
        # Increment view count
        item.view_count += 1
        db.session.commit()
        
        return jsonify({'item': item.to_dict()}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/items/<int:item_id>', methods=['PUT'])
@jwt_required()
def update_item(item_id):
    try:
        user_id = get_jwt_identity()
        item = Item.query.get_or_404(item_id)
        
        # Check if user owns the item
        if item.owner_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        
        # Update fields
        if 'title' in data:
            item.title = data['title']
        if 'description' in data:
            item.description = data['description']
        if 'brand' in data:
            item.brand = data['brand']
        if 'condition' in data:
            item.condition = data['condition']
        if 'estimated_value' in data:
            item.estimated_value = data['estimated_value']
        if 'is_available' in data:
            item.is_available = data['is_available']
        
        item.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Item updated successfully',
            'item': item.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/items/<int:item_id>', methods=['DELETE'])
@jwt_required()
def delete_item(item_id):
    try:
        user_id = get_jwt_identity()
        item = Item.query.get_or_404(item_id)
        
        # Check if user owns the item
        if item.owner_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Delete image file
        if os.path.exists(item.image_path):
            os.remove(item.image_path)
        
        db.session.delete(item)
        db.session.commit()
        
        return jsonify({'message': 'Item deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Categories routes
@app.route('/api/categories', methods=['GET'])
def get_categories():
    try:
        categories = Category.query.filter_by(is_active=True).all()
        return jsonify({
            'categories': [category.to_dict() for category in categories]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Brands routes
@app.route('/api/brands', methods=['GET'])
def get_brands():
    try:
        category = request.args.get('category')
        query = Brand.query.filter_by(is_active=True)
        
        if category:
            query = query.filter_by(category=category)
        
        brands = query.all()
        return jsonify({
            'brands': [brand.to_dict() for brand in brands]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Trades routes
@app.route('/api/trades', methods=['GET'])
@jwt_required()
def get_trades():
    try:
        user_id = get_jwt_identity()
        trade_type = request.args.get('type', 'all')  # sent, received, all
        
        if trade_type == 'sent':
            trades = Trade.query.filter_by(sender_id=user_id).all()
        elif trade_type == 'received':
            trades = Trade.query.filter_by(receiver_id=user_id).all()
        else:
            trades = Trade.query.filter(
                (Trade.sender_id == user_id) | (Trade.receiver_id == user_id)
            ).all()
        
        return jsonify({
            'trades': [trade.to_dict() for trade in trades]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades', methods=['POST'])
@jwt_required()
def create_trade():
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data.get('item_id') or not data.get('message'):
            return jsonify({'error': 'Item ID and message are required'}), 400
        
        item = Item.query.get_or_404(data['item_id'])
        
        # Check if user is not the owner
        if item.owner_id == user_id:
            return jsonify({'error': 'Cannot trade with yourself'}), 400
        
        # Check if item is available
        if not item.is_available:
            return jsonify({'error': 'Item is not available for trade'}), 400
        
        # Create new trade with additional info
        trade = Trade(
            item_id=item.id,
            sender_id=user_id,
            receiver_id=item.owner_id,
            message=data['message']
        )
        
        # Store sender's item info if provided
        if data.get('your_item_info'):
            trade.sender_item_info = data['your_item_info']
        
        db.session.add(trade)
        db.session.commit()
        
        return jsonify({
            'message': 'Trade request sent successfully',
            'trade': trade.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/api/trades/<int:trade_id>', methods=['PUT'])
@jwt_required()
def update_trade(trade_id):
    try:
        user_id = get_jwt_identity()
        trade = Trade.query.get_or_404(trade_id)
        
        # Check if user is involved in the trade
        if trade.sender_id != user_id and trade.receiver_id != user_id:
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        
        # Update status
        if 'status' in data:
            if data['status'] in ['accepted', 'rejected', 'completed', 'cancelled']:
                trade.status = data['status']
            else:
                return jsonify({'error': 'Invalid status'}), 400
        
        # Update other fields
        if 'message' in data:
            trade.message = data['message']
        if 'counter_offer' in data:
            trade.counter_offer = data['counter_offer']
        if 'meeting_location' in data:
            trade.meeting_location = data['meeting_location']
        if 'meeting_date' in data:
            trade.meeting_date = datetime.fromisoformat(data['meeting_date'])
        
        trade.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': 'Trade updated successfully',
            'trade': trade.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# AI Analysis routes
@app.route('/api/ai/analyze', methods=['POST'])
def analyze_item():
    try:
        # Demo mode - no authentication required
        user_id = None
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        try:
            # Use AI to analyze the image
            result = efficientnet_detector.analyze_image(temp_path)
            
            # Save analysis to database
            analysis = AIAnalysis(
                image_path=temp_path,
                brand_detected=result.get('brand_detected'),
                confidence_score=result.get('confidence'),
                model_used=result.get('model_used', 'EfficientNet-B3'),
                analysis_data=result,
                processing_time=result.get('processing_time', 0)
            )
            
            db.session.add(analysis)
            db.session.commit()
            
            return jsonify({
                'success': True,
                'analysis': analysis.to_dict(),
                'result': result,
                'category_detected': result.get('category_detected'),
                'brand_detected': result.get('brand_detected'),
                'confidence': result.get('confidence'),
                'description': result.get('description'),
                'predicted_price': result.get('predicted_price')
            }), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vlm/status', methods=['GET'])
def vlm_status():
    """Check VLM model status"""
    try:
        from efficientnet_detector import efficientnet_detector
        
        status = {
            'model_loaded': efficientnet_detector.model is not None,
            'price_model_loaded': efficientnet_detector.price_model is not None,
            'model_path': efficientnet_detector.model_path,
            'price_model_path': efficientnet_detector.price_model_path
        }
        
        return jsonify({
            'success': True,
            'status': status
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/debug/test-image', methods=['POST'])
def debug_test_image():
    """Debug endpoint to test image analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"debug_{filename}")
        file.save(filepath)
        
        # Test analysis
        result = efficientnet_detector.analyze_image(filepath)
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': True,
            'result': result,
            'model_loaded': efficientnet_detector.model is not None,
            'class_names_count': len(efficientnet_detector.class_names) if hasattr(efficientnet_detector, 'class_names') else 0
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/vlm/detect-brand', methods=['POST'])
def detect_brand():
    """EfficientNet-B3 endpoint to detect brand/logo from uploaded image"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Get model path from request (optional)
        model_path = request.form.get('model_path', None)
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        try:
            # Use EfficientNet-B3 to detect brand
            result = efficientnet_detector.analyze_image(temp_path)
            
            if result.get('success'):
                return jsonify({
                    'success': True,
                    'brand_detected': result.get('brand_detected'),
                    'confidence': result.get('confidence', 0.0),
                    'text_regions': result.get('text_regions', 0),
                    'logo_regions': result.get('logo_regions', 0),
                    'extracted_texts': result.get('extracted_texts', []),
                    'processing_time': result.get('processing_time', 'EfficientNet-B3 inference'),
                    'model_used': result.get('model_used', 'EfficientNet-B3'),
                    'top_3_predictions': result.get('top_3_predictions', []),
                    'message': f"Brand detected: {result.get('brand_detected', 'Unknown')}" if result.get('brand_detected') else "No brand detected in image"
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Brand detection failed'),
                    'brand_detected': None,
                    'confidence': 0.0,
                    'model_used': result.get('model_used', 'None')
                }), 400
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Brand detection error: {str(e)}',
            'brand_detected': None,
            'confidence': 0.0,
            'model_used': 'None'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/blip/generate-caption', methods=['POST'])
@jwt_required()
def generate_caption():
    """BLIP endpoint to generate image caption"""
    try:
        user_id = get_jwt_identity()
        
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{uuid.uuid4()}_{filename}")
        file.save(temp_path)
        
        try:
            # Use BLIP to generate caption
            caption = generate_image_caption(temp_path)
            
            return jsonify({
                'success': True,
                'caption': caption.get('caption', ''),
                'model_used': 'BLIP',
                'processing_time': 'CPU inference'
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Caption generation error: {str(e)}',
            'caption': ''
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(debug=debug, host='0.0.0.0', port=port)
