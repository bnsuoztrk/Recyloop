import cv2
import numpy as np
from PIL import Image
import os
import logging
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import json
import pickle
import pandas as pd
import lightgbm as lgb
import re

# Global VLM models for brand detection
try:
    from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
    import torch
    
    # Load models once at startup
    print("Loading CLIP and BLIP models for brand detection...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # Set to evaluation mode
    clip_model.eval()
    blip_model.eval()
    
    print("VLM models loaded successfully!")
    VLM_MODELS_LOADED = True
except Exception as e:
    print(f"Failed to load VLM models: {e}")
    clip_model = None
    clip_processor = None
    blip_model = None
    blip_processor = None
    VLM_MODELS_LOADED = False
# import torch
# import clip

# Import focal loss and classes - define locally to avoid circular import
import tensorflow as tf

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_crossentropy(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_crossentropy

# Classes array
CLASSES = [
    "babet", "bluz", "bot", "cizme", "cuzdan", "deri-ceket", "elbise", "etek", "gomlek", 
    "kaban", "kazak", "kol-cantasi", "kot-ceket", "laptop-cantasi", "mont", "pantolon", 
    "sandalet", "sirt-cantasi", "sort", "spor-ayakkabi", "stiletto", "sweatshirt", "terlik", "tshirt"
]

def preprocess_image_efficientnet(image_path: str):
    """EfficientNet preprocessing function"""
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    img = image.load_img(image_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EfficientNetLogoDetector:
    def __init__(self, model_path: str = None):
        """Initialize the EfficientNet-B3 logo detection system"""
        self.model = None
        self.price_model = None
        # Ensemble price prediction models
        self.lgb_model = None
        self.tokenizer = None
        self.scaler = None
        self.cat_encoders = None
        
        # Get absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(base_dir, "models", "efficientnetb3_prod_final.h5")
        self.price_model_path = os.path.join(base_dir, "models", "bilstm_full_model.keras")
        
        # Ensemble model paths
        self.tokenizer_path = os.path.join(base_dir, "models", "tokenizer.json")
        self.scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
        self.encoders_path = os.path.join(base_dir, "models", "cat_encoders.pkl")
        self.lgb_model_path = os.path.join(base_dir, "models", "lgb_model.txt")
        
        # Ensemble weights
        self.WEIGHT_LSTM = 0.7
        self.WEIGHT_LGBM = 0.3
        
        # Debug: Print paths
        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Price model path: {self.price_model_path}")
        logger.info(f"Model exists: {os.path.exists(self.model_path)}")
        logger.info(f"Price model exists: {os.path.exists(self.price_model_path)}")
        self.input_size = (300, 300)  # EfficientNet-B3 input size
        self.class_names = []
        
        # Description generator removed - using simple rule-based system
        
        
        # Initialize models if paths are provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif os.path.exists(self.model_path):
            try:
                self.load_model(self.model_path)
            except Exception as e:
                logger.error(f"Failed to load EfficientNet model: {e}")
                logger.warning("Continuing without EfficientNet model - only price prediction will work")
        else:
            logger.warning(f"Model path not found: {self.model_path}. No fallback detection.")
        
        if os.path.exists(self.price_model_path):
            self.load_price_model()
        else:
            logger.warning(f"Price model path not found: {self.price_model_path}")
            
        logger.info("EfficientNet detector initialized successfully")

    def load_model(self, model_path: str) -> bool:
        """Load the trained EfficientNet-B3 model with custom focal loss"""
        try:
            logger.info(f"Loading EfficientNet-B3 model from: {model_path}")
            
            # Test dosyasındaki çalışan kod - compile=False demiyoruz
            self.model = tf.keras.models.load_model(
                model_path,
                custom_objects={"focal_crossentropy": focal_loss(gamma=2.0, alpha=0.25)}
            )
            logger.info("Model loaded with custom focal loss!")
            
            logger.info("Model loaded successfully!")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            
            # Use the CLASSES array from app.py
            self.class_names = CLASSES
            logger.info(f"Using CLASSES from app.py: {len(self.class_names)} classes")
            logger.info(f"First 5 classes: {self.class_names[:5]}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
            return False

    def load_price_model(self) -> bool:
        """Load the price prediction model"""
        try:
            logger.info(f"Loading price model from: {self.price_model_path}")
            
            # Load the price model
            self.price_model = keras.models.load_model(self.price_model_path, compile=False)
            
            # Compile the model for inference
            self.price_model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            # Test model input/output format
            logger.info(f"Price model input shape: {self.price_model.input_shape}")
            logger.info(f"Price model output shape: {self.price_model.output_shape}")
            logger.info(f"Price model summary:")
            self.price_model.summary(print_fn=logger.info)
            
            logger.info("Price model loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading price model: {e}")
            self.price_model = None
            return False
    
    def load_ensemble_models(self):
        """Load ensemble price prediction models (BiLSTM + LightGBM)"""
        try:
            # Check if all ensemble model files exist
            ensemble_files = [
                self.tokenizer_path,
                self.scaler_path, 
                self.encoders_path,
                self.lgb_model_path
            ]
            
            missing_files = [f for f in ensemble_files if not os.path.exists(f)]
            if missing_files:
                logger.warning(f"Missing ensemble model files: {missing_files}")
                logger.warning("Falling back to simple BiLSTM model only")
                return False
            
            logger.info("Loading ensemble price prediction models...")
            
            # Load tokenizer
            with open(self.tokenizer_path, "r", encoding="utf-8") as f:
                tokenizer_data = json.load(f)
            self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_data)
            
            # Load scaler
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            
            # Load categorical encoders
            with open(self.encoders_path, "rb") as f:
                self.cat_encoders = pickle.load(f)
            
            # Load LightGBM model
            self.lgb_model = lgb.Booster(model_file=self.lgb_model_path)
            
            logger.info("✅ Ensemble models loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error loading ensemble models: {e}")
            logger.warning("Falling back to simple BiLSTM model only")
            return False
    
    def clean_text(self, text):
        """Clean text for tokenization"""
        text = str(text).lower()
        text = re.sub(r"[^a-z0-9ğüşıöç\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def build_text_input(self, data_dict):
        """Build text input from data dictionary"""
        text_parts = []
        for key in ["brand", "description", "subcategory", "main_category"]:
            if key in data_dict and data_dict[key]:
                text_parts.append(str(data_dict[key]))
        return self.clean_text(" ".join(text_parts))

    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Preprocess image for EfficientNet-B3 using correct preprocessing (test dosyasından)"""
        try:
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.applications.efficientnet import preprocess_input
            
            # Test dosyasındaki çalışan kod
            img = image.load_img(image_path, target_size=self.input_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)  # EfficientNet için özel preprocessing
            
            logger.info(f"Image preprocessed successfully: {img_array.shape}")
            return img_array
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def predict_price(self, category: str, confidence: float, brand: str = None, description: str = None) -> float:
        """Predict price using BiLSTM + LightGBM ensemble model"""
        try:
            # Don't predict price if category is not detected
            if category == "Sonuç Bulunamadı":
                logger.info("Category not detected - returning 0 for price")
                return 0.0
            
            # Try ensemble prediction first
            if self.lgb_model and self.tokenizer and self.scaler and self.cat_encoders:
                return self._predict_price_ensemble(category, confidence, brand, description)
            
            # Fallback to simple BiLSTM if ensemble models not available
            if self.price_model is None:
                logger.warning("No price models loaded - returning 0")
                return 0.0
            
            logger.info("Using simple BiLSTM model for price prediction")
            return self._predict_price_simple(category, confidence)
            
        except Exception as e:
            logger.error(f"Error in price prediction: {e}")
            return 0.0
    
    def _predict_price_ensemble(self, category: str, confidence: float, brand: str = None, description: str = None) -> float:
        """Predict price using BiLSTM + LightGBM ensemble"""
        try:
            logger.info("Using ensemble model for price prediction")
            
            # Prepare data dictionary
            data = {
                "brand": brand or "Unknown",
                "main_category": self._get_main_category(category),
                "subcategory": category,
                "description": description or f"{category} ürünü"
            }
            
            # Build text input
            text_input = self.build_text_input(data)
            seq = self.tokenizer.texts_to_sequences([text_input])
            X_pad = pad_sequences(seq, maxlen=self.price_model.input_shape[0][1], padding="post")
            
            # Prepare categorical inputs
            cat_inputs = []
            lgb_input = {}
            
            for col, encoder in self.cat_encoders.items():
                if col in data:
                    val = str(data[col])
                    if val in encoder.classes_:
                        encoded_val = encoder.transform([val])[0]
                    else:
                        encoded_val = 0
                else:
                    encoded_val = 0
                cat_inputs.append(np.array([encoded_val]))
                lgb_input[col] = encoded_val
            
            # Prepare numerical input (confidence)
            num_arr = self.scaler.transform([[confidence]])
            
            # BiLSTM prediction
            lstm_inputs = [X_pad] + cat_inputs + [num_arr]
            pred_lstm = np.expm1(self.price_model.predict(lstm_inputs, verbose=0))[0][0]
            
            # LightGBM prediction
            expected_features = self.lgb_model.feature_name()
            for feat in expected_features:
                if feat not in lgb_input:
                    lgb_input[feat] = 0
            
            df_lgb = pd.DataFrame([[lgb_input[f] for f in expected_features]], columns=expected_features)
            pred_lgb = np.expm1(self.lgb_model.predict(df_lgb, predict_disable_shape_check=True))[0]
            
            # Ensemble prediction
            price_pred = self.WEIGHT_LSTM * pred_lstm + self.WEIGHT_LGBM * pred_lgb
            price_pred = max(0.0, round(price_pred, 2))
            
            logger.info(f"Ensemble price prediction: {price_pred} TL (LSTM: {pred_lstm:.2f}, LGB: {pred_lgb:.2f})")
            return price_pred
            
        except Exception as e:
            logger.error(f"Error in ensemble price prediction: {e}")
            return self._predict_price_simple(category, confidence)
    
    def _predict_price_simple(self, category: str, confidence: float) -> float:
        """Fallback simple BiLSTM price prediction"""
        try:
            logger.info("Using simple BiLSTM model for price prediction")
            
            # Convert category to numerical representation
            category_index = self.class_names.index(category) if category in self.class_names else 0
            
            # Try different input formats
            model_inputs = [
                np.array([[category_index, confidence]]),
                np.array([[category_index]]),
                np.array([[confidence]]),
                np.array([[category_index, confidence, 0, 0, 0]]),
            ]
            
            for i, model_input in enumerate(model_inputs):
                try:
                    price_prediction = self.price_model.predict(model_input, verbose=0)
                    
                    if hasattr(price_prediction, 'shape'):
                        if len(price_prediction.shape) == 1:
                            predicted_price = float(price_prediction[0])
                        elif len(price_prediction.shape) == 2:
                            predicted_price = float(price_prediction[0][0])
                        else:
                            predicted_price = float(price_prediction[0])
                    else:
                        predicted_price = float(price_prediction)
                    
                    predicted_price = max(0.0, predicted_price)
                    logger.info(f"Simple price prediction: {predicted_price} TL for category: {category}")
                    return predicted_price
                    
                except Exception as e:
                    logger.warning(f"Input format {i+1} failed: {e}")
                    continue
            
            logger.error("All input formats failed for simple price prediction")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error in simple price prediction: {e}")
            return 0.0
    
    def _get_main_category(self, subcategory: str) -> str:
        """Get main category from subcategory"""
        main_categories = {
            'babet': 'Ayakkabi', 'bot': 'Ayakkabi', 'cizme': 'Ayakkabi', 'sandalet': 'Ayakkabi',
            'spor-ayakkabi': 'Ayakkabi', 'stiletto': 'Ayakkabi', 'terlik': 'Ayakkabi',
            'bluz': 'Giyim', 'deri-ceket': 'Giyim', 'elbise': 'Giyim', 'etek': 'Giyim',
            'gomlek': 'Giyim', 'kaban': 'Giyim', 'kazak': 'Giyim', 'kot-ceket': 'Giyim',
            'mont': 'Giyim', 'pantolon': 'Giyim', 'sort': 'Giyim', 'sweatshirt': 'Giyim', 'tshirt': 'Giyim',
            'cuzdan': 'Canta', 'kol-cantasi': 'Canta', 'laptop-cantasi': 'Canta', 'sirt-cantasi': 'Canta'
        }
        return main_categories.get(subcategory, 'Giyim')
    
    def _clean_vlm_caption(self, caption: str) -> str:
        """Clean VLM caption to create professional product description"""
        if not caption:
            return ""
        
        # Remove common prefixes and suffixes
        prefixes_to_remove = [
            'a photo of', 'an image of', 'a picture of', 'a photograph of',
            'this is', 'this shows', 'this image shows', 'this picture shows',
            'product description:', 'describe this', 'what is this', 'product details:',
            'describe', 'details', 'product:', 'features', 'what color is it?', 
            'what material is it made of?', 'in detail', 'adidas', 'nike', 'puma'
        ]
        
        clean_caption = caption.lower().strip()
        
        # Remove prefixes
        for prefix in prefixes_to_remove:
            if clean_caption.startswith(prefix):
                clean_caption = clean_caption[len(prefix):].strip()
        
        # Remove question marks and excessive punctuation
        clean_caption = clean_caption.replace('?', '').replace('!', '')
        clean_caption = ' '.join(clean_caption.split())  # Remove extra spaces
        
        # Remove repetitive words completely
        words = clean_caption.split()
        unique_words = []
        seen_words = set()
        
        for word in words:
            # Clean word (remove punctuation)
            clean_word = word.strip('.,!?;:').lower()
            if clean_word and clean_word not in seen_words:
                unique_words.append(word)
                seen_words.add(clean_word)
        
        # Create clean description (max 4 words)
        if len(unique_words) > 4:
            unique_words = unique_words[:4]
        
        clean_caption = ' '.join(unique_words)
        
        # Capitalize first letter and add period
        if clean_caption:
            clean_caption = clean_caption[0].upper() + clean_caption[1:] + "."
        
        return clean_caption

    def predict_brand(self, image_path: str) -> Dict:
        """Predict brand using VLM (Vision Language Model) - BLIP"""
        try:
            logger.info("Using VLM brand detection")
            
            # Use VLM for brand detection
            from blip_captioner import BLIPCaptioner
            
            # Initialize BLIP if not already done
            if not hasattr(self, 'blip_captioner'):
                self.blip_captioner = BLIPCaptioner()
            
            # Generate brand-focused caption
            brand_prompts = [
                "What brand or logo is visible in this image?",
                "Identify the brand name or company logo in this image",
                "What is the brand of this product?",
                "Name the brand or manufacturer visible in this image"
            ]
            
            best_result = None
            best_confidence = 0.0
            
            for prompt in brand_prompts:
                try:
                    result = self.blip_captioner.generate_caption(image_path, prompt)
                    if result.get('success'):
                        caption = result.get('caption', '').lower()
                        
                        # Extract brand from caption
                        brand = self._extract_brand_from_caption(caption)
                        if brand:
                            confidence = self._calculate_brand_confidence(caption, brand)
                            if confidence > best_confidence:
                                best_confidence = confidence
                                best_result = {
                                    'brand': brand,
                                    'confidence': confidence,
                                    'caption': result.get('caption'),
                                    'prompt': prompt
                                }
                except Exception as e:
                    logger.warning(f"VLM prompt failed: {e}")
                    continue
            
            if best_result and best_confidence > 0.3:  # 30% confidence threshold
                return {
                    'success': True,
                    'brand_detected': best_result['brand'],
                    'confidence': best_confidence,
                    'caption': best_result['caption'],
                    'model_used': 'BLIP-VLM',
                    'processing_time': 'CPU inference',
                    'top_3_predictions': [{
                        'brand': best_result['brand'],
                        'confidence': best_confidence
                    }]
                }
            else:
                return {
                    'success': False,
                    'brand_detected': None,
                    'confidence': best_confidence,
                    'error': 'No brand detected with sufficient confidence',
                    'model_used': 'BLIP-VLM'
                }
            
        except Exception as e:
            logger.error(f"Error in VLM brand prediction: {e}")
            return {
                'success': False,
                'brand_detected': None,
                'confidence': 0.0,
                'error': str(e),
                'model_used': 'BLIP-VLM'
            }

    def _extract_brand_from_caption(self, caption: str) -> str:
        """Extract brand name from VLM caption"""
        # Common brand keywords
        brand_keywords = [
            'nike', 'adidas', 'puma', 'zara', 'h&m', 'uniqlo', 'levi\'s', 'calvin klein',
            'tommy hilfiger', 'ralph lauren', 'gucci', 'prada', 'louis vuitton', 'chanel',
            'versace', 'armani', 'dolce', 'gabbana', 'balenciaga', 'off-white', 'supreme',
            'apple', 'samsung', 'sony', 'lg', 'huawei', 'xiaomi', 'oneplus', 'google',
            'microsoft', 'dell', 'hp', 'lenovo', 'asus', 'acer', 'msi', 'razer'
        ]
        
        caption_lower = caption.lower()
        
        # Look for brand keywords
        for brand in brand_keywords:
            if brand in caption_lower:
                return brand.title()
        
        # Look for patterns like "brand: xyz" or "logo: xyz"
        import re
        patterns = [
            r'brand[:\s]+([a-zA-Z\s]+)',
            r'logo[:\s]+([a-zA-Z\s]+)',
            r'company[:\s]+([a-zA-Z\s]+)',
            r'manufacturer[:\s]+([a-zA-Z\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, caption_lower)
            if match:
                brand = match.group(1).strip()
                if len(brand) > 1 and len(brand) < 50:  # Reasonable brand name length
                    return brand.title()
        
        return None
    
    def _calculate_brand_confidence(self, caption: str, brand: str) -> float:
        """Calculate confidence score for brand detection"""
        caption_lower = caption.lower()
        brand_lower = brand.lower()
        
        confidence = 0.0
        
        # Direct brand mention
        if brand_lower in caption_lower:
            confidence += 0.4
        
        # Brand-related keywords
        brand_indicators = ['brand', 'logo', 'company', 'manufacturer', 'label', 'tag']
        for indicator in brand_indicators:
            if indicator in caption_lower:
                confidence += 0.1
        
        # Confidence keywords
        confidence_indicators = ['clearly', 'obviously', 'definitely', 'sure', 'certain']
        for indicator in confidence_indicators:
            if indicator in caption_lower:
                confidence += 0.1
        
        # Uncertainty keywords (reduce confidence)
        uncertainty_indicators = ['maybe', 'might', 'could be', 'possibly', 'unclear']
        for indicator in uncertainty_indicators:
            if indicator in caption_lower:
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))

    def get_trained_class_names(self, num_classes: int):
        """Get the actual class names you trained the model with"""
        # Senin eğittiğin class'lar
        trained_classes = [
            # Shoes (Ayakkabı) - 7 class
            'babet',           # 0
            'çizme',           # 1
            'bot',             # 2
            'stiletto',        # 3
            'spor_ayakkabı',   # 4
            'sandalet',        # 5
            'terlik',          # 6
            
            # Bags (Çanta) - 4 class
            'cüzdan',          # 7
            'sırt_çantası',    # 8
            'laptop_çantası',  # 9
            'kol_çantası',     # 10
            
            # Clothing (Giyim) - 13 class
            't-shirt',         # 11
            'sweatshirt',      # 12
            'şort',            # 13
            'pantolon',        # 14
            'mont',            # 15
            'kot_ceket',       # 16
            'kazak',           # 17
            'kaban',           # 18
            'gömlek',          # 19
            'etek',            # 20
            'elbise',          # 21
            'deri_ceket',      # 22
            'bluz',            # 23
        ]
        
        # No generic class names - use exact match
        if len(trained_classes) != num_classes:
            logger.error(f"Class count mismatch! Expected {num_classes}, got {len(trained_classes)}")
            raise ValueError("Class count mismatch - cannot proceed without exact match")
        
        return trained_classes

    def predict_category(self, image_path: str) -> Dict:
        """Predict category using EfficientNet-B3 model (if available)"""
        try:
            if self.model is None:
                return {
                    'success': False,
                    'category_detected': None,
                    'confidence': 0.0,
                    'error': 'EfficientNet model not loaded - H5 file is corrupted'
                }
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return {
                    'success': False,
                    'category_detected': None,
                    'confidence': 0.0,
                    'error': 'Image preprocessing failed'
                }
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            top_prediction_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][top_prediction_idx])
            
            # Get predicted category name
            if top_prediction_idx < len(self.class_names):
                predicted_category = self.class_names[top_prediction_idx]
            else:
                return {
                    'success': False,
                    'category_detected': None,
                    'confidence': confidence,
                    'error': 'Invalid category index'
                }
            
            return {
                'success': True,
                'category_detected': predicted_category,
                'confidence': confidence,
                'model_used': 'EfficientNet-B3'
            }
            
        except Exception as e:
            logger.error(f"Error in category prediction: {e}")
            return {
                'success': False,
                'category_detected': None,
                'confidence': 0.0,
                'error': str(e)
            }
    

    def generate_description(self, image_path: str, category: str = None, brand: str = None) -> str:
        """Generate simple description based on brand and category"""
        try:
            logger.info(f"Generating description - Brand: {brand}, Category: {category}")
            
            # Simple rule-based description
            if brand and category:
                result = f"{brand} markası {category} ürünü"
            elif brand:
                result = f"{brand} markası kaliteli ürün"
            elif category:
                result = f"{category} kategorisinde ürün"
            else:
                result = "Görsel analiz edildi"
            
            logger.info(f"Generated description: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating description: {e}")
            return "Görsel analiz edildi."
    
    # Fallback description function removed - no default values
    
    def _clean_description(self, caption: str) -> str:
        """Clean and format description"""
        if not caption:
            return ""
        
        # Remove common prefixes
        prefixes_to_remove = [
            'a photo of', 'an image of', 'a picture of', 'a photograph of',
            'this is', 'this shows', 'this image shows', 'this picture shows'
        ]
        
        clean_caption = caption.lower().strip()
        
        for prefix in prefixes_to_remove:
            if clean_caption.startswith(prefix):
                clean_caption = clean_caption[len(prefix):].strip()
        
        # Capitalize first letter
        if clean_caption:
            clean_caption = clean_caption[0].upper() + clean_caption[1:]
        
        return clean_caption
    
    def _calculate_description_quality(self, description: str) -> float:
        """Calculate quality score for description"""
        if not description:
            return 0.0
        
        score = 0.0
        desc_lower = description.lower()
        
        # Length bonus (not too short, not too long)
        length = len(description.split())
        if 3 <= length <= 20:
            score += 0.3
        elif length > 20:
            score += 0.2
        
        # No default keywords - rely only on VLM analysis
        
        # No default generic phrases - rely only on VLM analysis
        
        return max(0.0, min(1.0, score))

    def detect_text_regions(self, image_path: str) -> List[tuple]:
        """Detect text regions in image"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                if area > 100 and w > 20 and h > 10:
                    text_regions.append((x, y, w, h))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []

    def detect_logo_regions(self, image_path: str) -> List[tuple]:
        """Detect potential logo regions"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            logo_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                aspect_ratio = w / h if h > 0 else 0
                
                # Filter for logo-like regions
                if (1000 < area < 50000 and 
                    0.3 < aspect_ratio < 3.0 and 
                    w > 30 and h > 30):
                    logo_regions.append((x, y, w, h))
            
            return logo_regions
            
        except Exception as e:
            logger.error(f"Error detecting logo regions: {e}")
            return []

    def analyze_image(self, image_path: str) -> Dict:
        """Main function to analyze image and detect category only - test dosyasından kopyalandı"""
        try:
            logger.info(f"Analyzing image: {image_path}")
            
            # Test dosyasındaki çalışan kod
            if self.model is None:
                return {
                    'success': False,
                    'brand_detected': None,
                    'category_detected': None,
                    'confidence': 0.0,
                    'predicted_price': 0.0,
                    'description': 'Model yüklenmedi',
                    'error': 'Model not loaded'
                }
            
            # Preprocess image (test dosyasındaki gibi)
            from tensorflow.keras.preprocessing import image
            from tensorflow.keras.applications.efficientnet import preprocess_input
            
            img = image.load_img(image_path, target_size=(300, 300))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            logger.info(f"Predicted class: {predicted_class}, confidence: {confidence}")
            
            # Check if confidence is too low
            if confidence < 0.3:  # 30% altında güven skoru
                logger.warning(f"Low confidence prediction: {confidence:.2f}")
                predicted_class = "Sonuç Bulunamadı"
                predicted_price = 0.0
                detected_brand = None
                description = "Görsel analiz edildi."
            else:
                # Detect brand using VLM
                detected_brand = self.detect_brand_with_vlm(image_path)
                
                # Generate description using VLM
                description = self._generate_description_with_vlm(image_path, predicted_class)
                
                # Generate price prediction with all available information
                predicted_price = self.predict_price(
                    category=predicted_class, 
                    confidence=confidence,
                    brand=detected_brand,
                    description=description
                )
            
            result = {
                'success': True,
                'brand_detected': detected_brand,
                'category_detected': predicted_class,
                'confidence': confidence,
                'predicted_price': predicted_price,
                'description': description,
                'model_used': 'EfficientNet-B3 + VLM'
            }
            
            logger.info(f"Analysis complete. Category: {result['category_detected']}")
            return result
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'brand_detected': None,
                'category_detected': None,
                'confidence': 0.0,
                'predicted_price': 0.0,
                'description': 'Görsel analiz edildi.',
                'error': str(e)
            }
    
    def detect_brand_with_vlm(self, image_path: str) -> str:
        """Detect brand using CLIP and BLIP models"""
        try:
            if not VLM_MODELS_LOADED:
                return None
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Try CLIP first for brand detection
            brand_texts = [
                "Nike logo", "Adidas logo", "Puma logo", "Reebok logo", "Converse logo",
                "Vans logo", "New Balance logo", "Under Armour logo", "Champion logo",
                "Lacoste logo", "Tommy Hilfiger logo", "Calvin Klein logo", "Levi's logo",
                "Zara logo", "H&M logo", "Uniqlo logo", "Gap logo", "Forever 21 logo",
                "Gucci logo", "Louis Vuitton logo", "Chanel logo", "Prada logo",
                "Versace logo", "Armani logo", "Hugo Boss logo", "Ralph Lauren logo"
            ]
            
            # CLIP brand detection
            inputs = clip_processor(text=brand_texts, images=image, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                outputs = clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                # Get top brand
                top_idx = probs.argmax().item()
                confidence = probs[0][top_idx].item()
                
                if confidence > 0.3:  # 30% confidence threshold
                    brand = brand_texts[top_idx].replace(" logo", "").replace("'s", "")
                    logger.info(f"CLIP detected brand: {brand} (confidence: {confidence:.3f})")
                    return brand
            
            # If CLIP fails, try BLIP for general brand detection
            inputs = blip_processor(image, "What brand or logo is visible in this image?", return_tensors="pt")
            
            with torch.no_grad():
                out = blip_model.generate(**inputs, max_length=50, num_beams=3)
                caption = blip_processor.decode(out[0], skip_special_tokens=True)
                
                # Extract brand from caption
                brand = self._extract_brand_from_caption(caption)
                if brand:
                    logger.info(f"BLIP detected brand: {brand}")
                    return brand
            
            return None
            
        except Exception as e:
            logger.error(f"Error in VLM brand detection: {e}")
            return None
    
    def _extract_brand_from_caption(self, caption: str) -> str:
        """Extract brand name from BLIP caption"""
        try:
            caption_lower = caption.lower()
            
            # Common brand names
            brands = [
                "nike", "adidas", "puma", "reebok", "converse", "vans", "new balance",
                "under armour", "champion", "lacoste", "tommy hilfiger", "calvin klein",
                "levi's", "zara", "h&m", "uniqlo", "gap", "forever 21", "gucci",
                "louis vuitton", "chanel", "prada", "versace", "armani", "hugo boss",
                "ralph lauren", "apple", "samsung", "sony", "lg", "huawei", "xiaomi"
            ]
            
            for brand in brands:
                if brand in caption_lower:
                    return brand.title()
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting brand from caption: {e}")
            return None

    def _generate_description_with_vlm(self, image_path: str, category: str) -> str:
        """Generate description using VLM analysis of the actual image"""
        try:
            if not VLM_MODELS_LOADED:
                return f"{category} kategorisinde ürün"
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Create simple, direct prompts for product description
            prompts = [
                f"Describe {category}",
                f"{category} details",
                f"Product: {category}",
                f"{category} features"
            ]
            
            descriptions = []
            
            # Get descriptions from BLIP
            for prompt in prompts:
                try:
                    inputs = blip_processor(image, prompt, return_tensors="pt")
                    
                    with torch.no_grad():
                        out = blip_model.generate(
                            **inputs, 
                            max_length=15, 
                            num_beams=3, 
                            do_sample=True, 
                            temperature=0.8,
                            repetition_penalty=3.0,
                            no_repeat_ngram_size=2,
                            early_stopping=True,
                            pad_token_id=blip_processor.tokenizer.eos_token_id
                        )
                        caption = blip_processor.decode(out[0], skip_special_tokens=True)
                        # Clean the caption
                        caption = self._clean_vlm_caption(caption)
                        descriptions.append(caption)
                        
                except Exception as e:
                    logger.warning(f"VLM prompt failed: {e}")
                    continue
            
            if descriptions:
                # Combine and clean up descriptions
                combined_description = self._combine_descriptions(descriptions, category)
                logger.info(f"VLM generated description: {combined_description}")
                return combined_description
            else:
                return f"{category} kategorisinde ürün"
                
        except Exception as e:
            logger.error(f"Error generating VLM description: {e}")
            return f"{category} kategorisinde ürün"
    
    # Translation functions removed - using VLM descriptions directly
    
    def _combine_descriptions(self, descriptions: list, category: str) -> str:
        """Combine multiple VLM descriptions into a coherent description"""
        try:
            # Extract key features from descriptions
            features = {
                'colors': [],
                'materials': [],
                'styles': [],
                'details': []
            }
            
            # No default word lists - use VLM descriptions directly
            for desc in descriptions:
                features['details'].append(desc)
            
            # Create professional product description
            if features['details']:
                # Get the best description (longest and most meaningful)
                best_description = ""
                for desc in features['details']:
                    # Check if description is meaningful (not just prompt text)
                    if (len(desc) > len(best_description) and 
                        len(desc.split()) >= 2 and 
                        not any(word in desc.lower() for word in ['describe', 'details', 'product:', 'features', 'what', 'color', 'material'])):
                        best_description = desc
                
                if best_description and len(best_description.strip()) > 3:
                    return best_description
                else:
                    # No meaningful description found - return empty for user input
                    return ""
            else:
                # No description found - return empty for user input
                return ""
                
        except Exception as e:
            logger.error(f"Error combining descriptions: {e}")
            return f"{category} kategorisinde ürün"

# Global instance with default model path - test dosyasındaki gibi
efficientnet_detector = EfficientNetLogoDetector()
# Model yüklendiğinden emin ol
if efficientnet_detector.model is None:
    print("Model yüklenmedi, yeniden yükleniyor...")
    efficientnet_detector.load_model(efficientnet_detector.model_path)

# Load ensemble models for advanced price prediction
print("Loading ensemble price prediction models...")
ensemble_loaded = efficientnet_detector.load_ensemble_models()
if ensemble_loaded:
    print("Ensemble models loaded successfully!")
else:
    print("Ensemble models not available, using simple BiLSTM only")

def detect_brand_from_image(image_path: str, model_path: str = None) -> Dict:
    """Public function to detect brand from image using EfficientNet-B3"""
    global efficientnet_detector
    
    # Load model if path is provided and different from current
    if model_path and model_path != efficientnet_detector.model_path:
        efficientnet_detector = EfficientNetLogoDetector(model_path)
    
    return efficientnet_detector.analyze_image(image_path)
