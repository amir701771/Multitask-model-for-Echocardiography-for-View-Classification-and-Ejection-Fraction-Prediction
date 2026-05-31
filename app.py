
import os
import logging
import torch
import numpy as np
import sqlite3
import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from torchvision import transforms as T
import cv2
import sys
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user, login_url
from flask_bcrypt import Bcrypt
from flask import flash, redirect, url_for

# Add src to path so we can import models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src.models.cnn_multitask import MultiTaskCNN
except ImportError:
    # Fallback/Debug if src issues
    from src.models.cnn_view_classifier import ViewClassifierCNN
    MultiTaskCNN = None

import threading
from src.training.pipeline_ui import run_training_pipeline

app = Flask(__name__)

# Global Training State
TRAINING_STATE = {
    "status": "idle", # idle, training, completed, failed, stopped
    "progress": 0,
    "epoch": 0,
    "total_epochs": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "message": ""
}
STOP_EVENT = threading.Event()
app.config['UPLOAD_FOLDER'] = 'static/uploads' # Changed to static so it's accessible
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
app.config['DB_PATH'] = 'echonet.db'
app.secret_key = 'super_secret_production_key_medical_ai_2025' # REQUIRED for session/flash
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Authentication Setup ---
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username=None):
        self.id = id
        self.username = username

@login_manager.user_loader
def user_loader(id):
    try:
        with sqlite3.connect(app.config['DB_PATH']) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username FROM users WHERE id = ?", (id,))
            user_data = cursor.fetchone()
            if user_data:
                return User(id=user_data[0], username=user_data[1])
            return None
    except Exception as e:
        logging.error(f"User Loader Error: {e}")
        return None

@login_manager.unauthorized_handler
def unauthorized():
    # Return JSON error for API calls, redirect for page loads
    if request.is_json or request.path.startswith('/history') or request.path.startswith('/api/'):
        return jsonify({'error': 'Unauthorized', 'success': False}), 401
    return redirect(url_for('login'))

# Logging
logging.basicConfig(level=logging.INFO)

# --- Database Setup ---
# --- Database Setup ---
def init_db():
    with sqlite3.connect(app.config['DB_PATH']) as conn:
        cursor = conn.cursor()
        # History Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                ef_value REAL,
                ef_category TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Users Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        
        try:
            cursor.execute('SELECT ef_value FROM history LIMIT 1')
        except sqlite3.OperationalError:
            logging.info("Migrating DB: Adding ef_value and ef_category columns.")
            cursor.execute('ALTER TABLE history ADD COLUMN ef_value REAL')
            cursor.execute('ALTER TABLE history ADD COLUMN ef_category TEXT')
            conn.commit()
        conn.commit()

init_db()

# --- Configuration & Model Loading ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Find Model
POSSIBLE_PATHS = [
    'results/view_classifier_echonet/checkpoints/best_model.pth.tar',
    'results/view_classifier_echonet/checkpoints/checkpoint.pth.tar',
]
MODEL_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        MODEL_PATH = p
        break

# Constants
# Constants
IMG_SIZE = (112, 112) # Match training size exactly
# STRICT BINARY MAPPING
VIEW_MAPPING = {0: 'A2C', 1: 'A4C'} 

model = None

def load_model():
    global model
    
    if MultiTaskCNN is None:
        logging.error("MultiTaskCNN class not found. Cannot proceed with EF prediction.")
        return

    # Initialize MultiTaskCNN (Views + EF)
    try:
        # STRICT REQUIREMENT: num_view_classes = 2 (A2C, A4C only)
        model = MultiTaskCNN(backbone_name='resnet18', num_view_classes=2).to(DEVICE)
    except Exception as e:
        logging.error(f"Failed to init MultiTaskCNN: {e}")
        return

    if not MODEL_PATH:
        logging.warning("No model file found! Model is using random initialization.")
        return

    try:
        logging.info(f"Loading weights from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Clean state dict keys (remove 'module.')
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Attempt to load whatever matches using strict=False
        missing, unexpected = model.load_state_dict(clean_state_dict, strict=False)
        logging.info(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # Handle View Classifier Weight Transfer if mismatch exists
        if 'classifier.weight' in clean_state_dict:
             src_weight = clean_state_dict['classifier.weight']
             src_bias = clean_state_dict['classifier.bias']
             
             if src_weight.shape == model.view_classifier.weight.shape:
                  with torch.no_grad():
                      model.view_classifier.weight.copy_(src_weight)
                      model.view_classifier.bias.copy_(src_bias)
                  logging.info("Transferred classifier weights (exact match).")
             elif src_weight.shape[1] == model.view_classifier.weight.shape[1]: 
                  min_classes = min(src_weight.shape[0], model.view_classifier.weight.shape[0])
                  with torch.no_grad():
                      model.view_classifier.weight[:min_classes] = src_weight[:min_classes]
                      model.view_classifier.bias[:min_classes] = src_bias[:min_classes]
                  logging.info(f"Transferred partial classifier weights ({min_classes} classes).")
        
        model.eval()
        logging.info(f"Model successfully loaded on {DEVICE}.")
        
    except Exception as e:
        logging.error(f"Failed to load weights: {e}", exc_info=True)
        model = None

load_model()

def get_inference_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])

def extract_frames(video_path, num_frames=16):
    """Uniformly sample 'num_frames' frames from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback for streams where frame count isn't available
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        return frames[:num_frames] if len(frames) > num_frames else frames

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    
    cap.release()
    return frames

def save_to_history(filename, label, confidence, ef_value=None, ef_category=None):
    try:
        with sqlite3.connect(app.config['DB_PATH']) as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO history 
                   (filename, prediction, confidence, ef_value, ef_category) 
                   VALUES (?, ?, ?, ?, ?)''',
                (filename, label, confidence, ef_value, ef_category)
            )
            conn.commit()
    except Exception as e:
        logging.error(f"DB Error: {e}")

# Video Conversion Utility
# Try to import imageio_ffmpeg, handled safely
try:
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    FFMPEG_EXE = None
    logging.warning("imageio-ffmpeg not found. Video conversion may fail if system ffmpeg is missing.")

import subprocess

def convert_to_mp4(input_path, output_path):
    """
    Converts a video to MP4 using ffmpeg (via imageio-ffmpeg or system path).
    Ensures H.264 video and AAC audio for maximum browser compatibility.
    """
    ffmpeg_cmd = FFMPEG_EXE or 'ffmpeg'
    
    # Base command: 
    # -y (overwrite)
    # -i input
    # -c:v libx264 (H.264 Video)
    # -preset fast (Fast encoding)
    # -crf 23 (Standard quality)
    # -c:a aac (AAC Audio)
    # -strict experimental (sometimes needed for older ffmpeg aac)
    # -movflags +faststart (Web optimization)
    # -pix_fmt yuv420p (Ensure compatibility with all players)
    
    cmd = [
        ffmpeg_cmd, '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23', 
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-movflags', '+faststart',
        output_path
    ]
    
    try:
        logging.info(f"Starting ffmpeg conversion: {' '.join(cmd)}")
        # Run subprocess, capturing output for debugging
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            logging.error(f"FFmpeg failed: {result.stderr}")
            return False
            
        logging.info(f"FFmpeg conversion successful: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Conversion processing error: {e}")
        return False

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if not username or not password:
             flash('Please fill in all fields')
             return redirect(url_for('register'))
             
        if password != confirm_password:
             flash('Passwords do not match')
             return redirect(url_for('register'))
             
        try:
            with sqlite3.connect(app.config['DB_PATH']) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                if cursor.fetchone():
                    flash('Username already exists')
                    return redirect(url_for('register'))
                
                hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
                cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
                conn.commit()
                flash('Registration successful! Please login.')
                return redirect(url_for('login'))
        except Exception as e:
            flash(f"Error: {e}")
            return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        try:
            with sqlite3.connect(app.config['DB_PATH']) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
                user_data = cursor.fetchone()
                
                if user_data and bcrypt.check_password_hash(user_data['password'], password):
                    user_obj = User(id=user_data['id'], username=user_data['username'])
                    login_user(user_obj)
                    return redirect(url_for('index'))
                else:
                    flash('Invalid username or password')
                    return redirect(url_for('login'))
        except Exception as e:
             logging.error(f"Login Error: {e}")
             flash('An error occurred during login')
            
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        # Save original
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        final_video_path = filepath
        final_filename = filename
        
        # Check if conversion needed (simple check by extension)
        if not filename.lower().endswith('.mp4'):
            mp4_filename = os.path.splitext(filename)[0] + '.mp4'
            mp4_path = os.path.join(app.config['UPLOAD_FOLDER'], mp4_filename)
            
            logging.info(f"Converting {filename} to {mp4_filename} for browser playback...")
            success = convert_to_mp4(filepath, mp4_path)
            
            if success:
                final_video_path = mp4_path
                final_filename = mp4_filename
            else:
                logging.warning("Conversion failed, falling back to original file.")
        
        # Generate URL for frontend
        video_url = f"/static/uploads/{final_filename}"
        
        if model is None:
            return jsonify({'error': 'Model not loaded', 'video_url': video_url}), 500
        
        try:
            # 1. Extract Frames
            frames = extract_frames(final_video_path, num_frames=16)
            if not frames:
                return jsonify({'error': 'Could not read video frames.', 'video_url': video_url}), 400
                
            # 2. Preprocess
            transform = get_inference_transforms()
            tensors = [transform(f) for f in frames]
            input_tensor = torch.stack(tensors).to(DEVICE) # [T, C, H, W]
            
            # 3. Inference
            with torch.no_grad():
                view_logits, ef_preds = model(input_tensor)
                
                # Average view logits
                avg_view_logits = view_logits.mean(dim=0)
                probs = torch.softmax(avg_view_logits, dim=0)
                conf, pred_idx_tensor = torch.max(probs, dim=0)
                pred_idx = pred_idx_tensor.item()
                confidence = conf.item()
                
                # Map Label
                view_label = VIEW_MAPPING.get(pred_idx, "Unknown")
                
                 # EF Logic - STRICT
                ef_val_scalar = None
                ef_category = None
                ef_message = "Not Applicable"
                
                if pred_idx == 1: # A4C (Class 1)
                     # EF is only valid for A4C
                     avg_ef = ef_preds.mean().item() # Already clamped 10-95 in model
                     
                     if avg_ef >= 55: cat = "Normal"
                     elif 40 <= avg_ef < 55: cat = "Mildly Reduced"
                     else: cat = "Reduced"
                     
                     ef_val_scalar = round(avg_ef, 1)
                     ef_category = cat
                     ef_message = f"{ef_val_scalar}% ({cat})"
                
                elif pred_idx == 0: # A2C (Class 0)
                     ef_val_scalar = None
                     ef_category = "N/A"
                     ef_message = "EF Not Applicable (A2C View)"
                
                else:
                     ef_val_scalar = None
                     ef_category = "Unknown View"
                     ef_message = "EF analysis requires A4C view"

            # 4. Save to History
            save_to_history(filename, view_label, confidence, ef_val_scalar, ef_category)
            
            return jsonify({
                'label': view_label,
                'confidence': f"{confidence*100:.1f}%",
                'filename': final_filename, 
                'video_url': video_url,     
                'ef_value': ef_val_scalar,
                'ef_category': ef_category,
                'ef_message': ef_message
            })
            
        except Exception as e:
            logging.error(f"Inference Error: {e}", exc_info=True)
            return jsonify({'error': str(e), 'video_url': video_url}), 500

@app.route('/history', methods=['GET'])
@login_required
def get_history():
    try:
        with sqlite3.connect(app.config['DB_PATH']) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM history ORDER BY timestamp DESC LIMIT 50')
            rows = cursor.fetchall()
            history_data = [dict(row) for row in rows]
            return jsonify(history_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<int:id>', methods=['DELETE'])
@login_required
def delete_history_item(id):
    try:
        logging.info(f"Received DELETE request for history ID: {id}")
        with sqlite3.connect(app.config['DB_PATH']) as conn:
            cursor = conn.cursor()
            
            # 1. Get filename to delete file from disk
            cursor.execute('SELECT filename FROM history WHERE id = ?', (id,))
            row = cursor.fetchone()
            
            if not row:
                logging.warning(f"Delete failed: History ID {id} not found.")
                return jsonify({'success': False, 'error': 'Record not found'}), 404
                
            filename = row[0]
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 2. Delete file from disk
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted file from disk: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to delete file {file_path}: {e}")
                    # Continue to delete DB record
            else:
                 logging.warning(f"File not found on disk: {file_path}")
                 
            # 3. Delete from DB
            cursor.execute('DELETE FROM history WHERE id = ?', (id,))
            conn.commit()
            
            if cursor.rowcount > 0:
                logging.info(f"Successfully deleted history record ID: {id}")
                return jsonify({'success': True, 'message': 'Analysis deleted successfully'})
            else:
                return jsonify({'success': False, 'error': 'Failed to delete record from DB'}), 500

    except Exception as e:
        logging.error(f"Delete Endpoint Error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/training-status', methods=['GET'])
@login_required
def get_training_status():
    if current_user.username != 'admin': # Simple check, better role system advised
         pass 
    return jsonify(TRAINING_STATE)

def update_training_status(status_dict):
    """Callback to update global state safely."""
    TRAINING_STATE.update(status_dict)

@app.route('/api/train-model', methods=['POST'])
@login_required
def start_training_model():
    # Basic Authorization check (assuming 'admin' username for demo)
    # in real app, check user.role or specific permission
    
    global TRAINING_STATE
    if TRAINING_STATE['status'] == 'training':
        return jsonify({'status': 'error', 'message': 'Training already in progress'}), 400
        
    TRAINING_STATE = {
        "status": "training",
        "progress": 0,
        "epoch": 0,
        "total_epochs": 0,
        "loss": 0.0,
        "accuracy": 0.0,
        "message": "Initializing..."
    }
    
    config_path = "config/config_echonet_multitask.yaml"
    
    def run_thread():
        run_training_pipeline(config_path, status_callback=update_training_status)
        # Reload model if successful
        if TRAINING_STATE['status'] == 'completed':
            logging.info("Training complete. Reloading model weights...")
            load_model()
            
    thread = threading.Thread(target=run_thread)
    thread.start()
    
    return jsonify({'status': 'training_started'})

@app.route('/uploads/<filename>')
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/train', methods=['POST'])
@login_required
def start_training_simple():
    """
    Simpler, more robust training endpoint.
    Returns immediately, runs in background.
    """
    global TRAINING_STATE
    
    # If already training, return 400 BUT don't crash or hang
    if TRAINING_STATE['status'] == 'training':
        return jsonify({'status': 'already_training', 'message': 'Training is already in progress'}), 400

    # Reset State for new run
    STOP_EVENT.clear()
    TRAINING_STATE = {
        "status": "training",
        "progress": 0,
        "epoch": 0,
        "total_epochs": 0,
        "loss": 0.0,
        "accuracy": 0.0,
        "message": "Initializing training process..."
    }
    
    config_path = "config/config_echonet_multitask.yaml"

    def run_safe_thread():
        try:
            # We must import here or ensure it's imported at top
            logging.info("Background thread started for training...")
            run_training_pipeline(config_path, status_callback=update_training_status, stop_flag_callback=STOP_EVENT.is_set)
            
            # If we reached here without exception from pipeline, check logic
            if TRAINING_STATE['status'] == 'completed':
                 logging.info("Background training completed successfully. Reloading model.")
                 with app.app_context(): # Ensure app context if load_model needs it (it doesn't currently, but safe practice)
                    load_model()
            
        except Exception as e:
            logging.error(f"Background Thread Crash: {e}", exc_info=True)
            update_training_status({
                "status": "failed", 
                "message": f"Server Side Error: {str(e)}"
            })

    # Daemon thread to ensure it doesn't block server shutdown if needed, 
    # but more importantly it runs detached from the request context.
    thread = threading.Thread(target=run_safe_thread, daemon=True)
    thread.start()
    
    return jsonify({'status': 'started', 'message': 'Training background process started'})

@app.route('/api/stop-training', methods=['POST'])
@login_required
def stop_training():
    """
    Stop the currently running training process gracefully.
    """
    if TRAINING_STATE['status'] != 'training':
        return jsonify({'success': False, 'message': 'No training is currently running'}), 400
    
    STOP_EVENT.set()
    logging.info("Training stop requested by user")
    
    return jsonify({'success': True, 'message': 'Stop signal sent to training process'})

if __name__ == '__main__':
    # Threaded=True is critical for handling the background thread + new requests
    app.run(debug=True, use_reloader=False, threaded=True)
