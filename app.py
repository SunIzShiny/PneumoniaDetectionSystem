import os
import time
import sqlite3
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.utils import secure_filename
from predict_gradcam import predict_all
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from scipy.spatial.distance import cosine


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

TEMPLATE_FOLDER = os.path.join(PROJECT_ROOT, "templates")
STATIC_FOLDER = os.path.join(PROJECT_ROOT, "static")
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, "uploads")
DB_FOLDER = os.path.join(PROJECT_ROOT, "db")
DATABASE = os.path.join(DB_FOLDER, "pneumonia.db")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "supersecretkey"


def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

conn = get_db_connection()

conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    role TEXT NOT NULL
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    patient_name TEXT,
    patient_id TEXT,
    dob TEXT,
    image_path TEXT,
    label TEXT,
    UNIQUE(user_id, patient_id)
)
""")

conn.commit()

cur = conn.execute("SELECT * FROM users WHERE username=?", ('admin',))
if not cur.fetchone():
    conn.execute(
        "INSERT INTO users (username,password,role) VALUES (?,?,?)",
        ('admin', 'admin123', 'admin')
    )
conn.commit()
conn.close()


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_logged_in():
    return 'user_id' in session

def is_admin():
    return session.get('role') == 'admin'


resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


FEATURES_FILE = os.path.join(PROJECT_ROOT, "chest_features.npy")
if os.path.exists(FEATURES_FILE):
    chest_features = np.load(FEATURES_FILE)
else:
    
    chest_dir = os.path.join(PROJECT_ROOT, "archive","chest_xray","chest_xray" ,"train")
    features_list = []
    for subfolder in ['NORMAL', 'PNEUMONIA']:
        folder_path = os.path.join(chest_dir, subfolder)
        for fname in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, fname)
                img = image.load_img(img_path, target_size=(224, 224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                feat = resnet_model.predict(x)
                features_list.append(feat[0])
            except:
                continue
    chest_features = np.stack(features_list)
    np.save(FEATURES_FILE, chest_features)

NON_CHEST_FEATURES_FILE = os.path.join(PROJECT_ROOT, "non_chest_features.npy")
if os.path.exists(NON_CHEST_FEATURES_FILE):
    non_chest_features = np.load(NON_CHEST_FEATURES_FILE)
else:
    non_chest_dir = os.path.join(PROJECT_ROOT, "archive", "non_chest_xray")
    non_features_list = []
    for fname in os.listdir(non_chest_dir):
        try:
            img_path = os.path.join(non_chest_dir, fname)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = resnet_model.predict(x)
            non_features_list.append(feat[0])
        except:
            continue
    non_chest_features = np.stack(non_features_list)
    np.save(NON_CHEST_FEATURES_FILE, non_chest_features)


def is_chest_xray(upload_path, chest_threshold=0.5):
    """
    Returns True only if the uploaded image is closer to chest X-rays than non-chest X-rays.
    """
    try:
        img = image.load_img(upload_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = resnet_model.predict(x)[0]

        
        chest_sims = [1 - cosine(feat, cf) for cf in chest_features]
        max_chest_sim = max(chest_sims)

        
        non_sims = [1 - cosine(feat, cf) for cf in non_chest_features]
        max_non_sim = max(non_sims)

        
        if max_chest_sim >= chest_threshold and max_chest_sim > max_non_sim:
            return True
        return False
    except:
        return False




@app.route('/', methods=['GET', 'POST'])
def home():
    filename = None
    result = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and allowed_file(file.filename):
            filename = f"{int(time.time())}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            
            if not is_chest_xray(filepath):
                flash("Invalid image! Please upload a chest X-ray.", "danger")
                os.remove(filepath)
                return redirect(url_for('home'))

            
            result = predict_all(filepath)
        else:
            flash("Only image files allowed!", "danger")

    return render_template("home.html", filename=filename, result=result)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        if not username or not password:
            flash("Please fill all fields.", "danger")
            return redirect(url_for('signup'))

        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (username,password,role) VALUES (?,?,?)",
                (username, password, 'user')
            )
            conn.commit()
            flash("Signup successful! You can now login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
        finally:
            conn.close()

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()

        conn = get_db_connection()
        user = conn.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (username, password)
        ).fetchone()
        conn.close()

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']

            flash("Login successful!", "success")
            return redirect(url_for('admin_dashboard' if user['role'] == 'admin' else 'dashboard'))

        flash("Invalid credentials!", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for('home'))


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if not is_logged_in() or is_admin():
        flash("Access denied.", "danger")
        return redirect(url_for('login'))

    filename = None
    result = None

    if request.method == 'POST':
        patient_name = request.form.get('patient_name', '').strip()
        patient_id = request.form.get('patient_id', '').strip()
        dob = request.form.get('dob', '').strip()
        file = request.files.get('file')

        if not all([patient_name, patient_id, dob, file]):
            flash("Please fill all fields and upload an image.", "danger")
            return redirect(url_for('dashboard'))

        if not allowed_file(file.filename):
            flash("Invalid image format!", "danger")
            return redirect(url_for('dashboard'))

        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        
        conn = get_db_connection()
        existing = conn.execute(
            "SELECT * FROM history WHERE user_id=? AND patient_id=?",
            (session['user_id'], patient_id)
        ).fetchone()

        if existing:
            flash(f"Patient ID '{patient_id}' already exists for you!", "danger")
            conn.close()
            return redirect(url_for('dashboard'))

       
        if not is_chest_xray(filepath):
            flash("Invalid image! Please upload a chest X-ray.", "danger")
            os.remove(filepath)
            conn.close()
            return redirect(url_for('dashboard'))

        
        result = predict_all(filepath)
        final_label = result["label"]

        
        conn.execute("""
            INSERT INTO history (user_id, patient_name, patient_id, dob, image_path, label)
            VALUES (?,?,?,?,?,?)
        """, (
            session['user_id'],
            patient_name,
            patient_id,
            dob,
            filename,
            final_label
        ))
        conn.commit()
        conn.close()

        flash("Prediction successful!", "success")

    conn = get_db_connection()
    records = conn.execute(
        "SELECT * FROM history WHERE user_id=?",
        (session['user_id'],)
    ).fetchall()
    conn.close()

    return render_template(
        "dashboard.html",
        filename=filename,
        result=result,
        records=records
    )


@app.route('/history')
def history():
    if not is_logged_in() or is_admin():
        flash("Access denied.", "danger")
        return redirect(url_for('login'))

    conn = get_db_connection()
    records = conn.execute(
        "SELECT * FROM history WHERE user_id=?",
        (session['user_id'],)
    ).fetchall()
    conn.close()

    return render_template('history.html', records=records)

@app.route('/delete_history/<int:record_id>')
def delete_history(record_id):
    if not is_logged_in() or is_admin():
        flash("Access denied.", "danger")
        return redirect(url_for('login'))

    conn = get_db_connection()
    record = conn.execute(
        "SELECT * FROM history WHERE id=? AND user_id=?",
        (record_id, session['user_id'])
    ).fetchone()

    if record:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], record['image_path'])
        if os.path.exists(img_path):
            os.remove(img_path)

        conn.execute("DELETE FROM history WHERE id=?", (record_id,))
        conn.commit()

    conn.close()
    flash("Record deleted.", "success")
    return redirect(url_for('history'))


@app.route('/admin')
def admin_dashboard():
    if not is_logged_in() or not is_admin():
        flash("Admin access only.", "danger")
        return redirect(url_for('login'))

    conn = get_db_connection()
    users = conn.execute(
        "SELECT * FROM users WHERE role='user'"
    ).fetchall()
    conn.close()

    return render_template('admin_dashboard.html', users=users)

@app.route('/admin/delete_user/<int:user_id>')
def admin_delete_user(user_id):
    if not is_logged_in() or not is_admin():
        flash("Admin access only.", "danger")
        return redirect(url_for('login'))

    conn = get_db_connection()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.execute("DELETE FROM history WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

    flash("User deleted.", "success")
    return redirect(url_for('admin_dashboard'))


@app.route('/uploads/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
