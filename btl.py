from flask import Flask, request, render_template
import pandas as pd
import csv
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# globals lưu mô hình và kết quả accuracy
trained_models = {}
vectorizer = None
label_encoder = None
accuracy_results = {}

def clean_text(text): # Chuẩn hóa văn bản  loại bỏ ký tự đặc biệt, chuyển về chữ thường, loại bỏ khoảng trắng thừa
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def assign_label(text):
    keyword_labels = {
        'robot': 'robotics',
        'navigation': 'robotics',
        'depth': 'vision',
        'stereo': 'vision',
        'image': 'vision',
        'text': 'NLP',
        'classification': 'NLP',
        'language': 'NLP'
    }
    s = text.lower()
    for kw, lbl in keyword_labels.items():
        if kw in s:
            return lbl
    return None

@app.route('/', methods=['GET'])
def index():
    return render_template(
        'index.html',
        acc_knn=accuracy_results.get('knn'),
        acc_id3=accuracy_results.get('id3'),
        acc_nb=accuracy_results.get('nb'),
    )

@app.route('/run-models', methods=['POST'])
def run_models():
    global trained_models, vectorizer, label_encoder, accuracy_results

    f = request.files.get('file')
    if not f:
        return render_template('index.html', error="Bạn chưa chọn file!")

    try:
        df = pd.read_csv(
            f,
            engine='python',
            sep=',',
            quotechar='"',
            quoting=csv.QUOTE_NONE,
            escapechar='\\',
            on_bad_lines='skip',
            encoding='utf-8'
        )
    except Exception as e:
        return render_template('index.html', error=f"Lỗi đọc CSV: {e}")

    # Gộp văn bản
    df['text'] = (
        df['titles'].fillna('') + ' ' +
        df['summaries'].fillna('') + ' ' +
        df['terms'].fillna('')
    )

    # Làm sạch và lọc
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.split().str.len() > 10]

    # Gán nhãn và loại bỏ None
    df['label'] = df['text'].apply(assign_label)
    df = df.dropna(subset=['label'])

    # Vector hóa và mã hóa label
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=2) 
    # Loại bỏ từ dừng thông dụng bằng stop_words='english' và dùng TfidfVectorizer để biểu diễn văn bản dưới dạng ma trận TF-IDF (tách từ)
    X = vectorizer.fit_transform(df['text']) # Biến đổi danh sách văn bản thành ma trận TF–IDF (chuyển chuổi ký tự thành vecto số)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Huấn luyện
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=10),
        "Naive Bayes": MultinomialNB()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = accuracy_score(y_test, y_pred)

    # Lưu mô hình và accuracy
    trained_models = models
    accuracy_results = {
        'knn': f"{results['KNN']:.4f}",
        'id3': f"{results['Decision Tree']:.4f}",
        'nb':  f"{results['Naive Bayes']:.4f}"
    }

    return render_template(
        'index.html',
        acc_knn=accuracy_results['knn'],
        acc_id3=accuracy_results['id3'],
        acc_nb=accuracy_results['nb']
    )

@app.route('/predict', methods=['POST'])
def predict():
    global trained_models, vectorizer, label_encoder, accuracy_results

    input_text = request.form.get('test_text')
    if not input_text:
        return render_template(
            'index.html',
            error="Bạn chưa nhập văn bản!",
            acc_knn=accuracy_results.get('knn'),
            acc_id3=accuracy_results.get('id3'),
            acc_nb=accuracy_results.get('nb')
        )

    if not trained_models:
        return render_template(
            'index.html',
            error="Chưa có mô hình để dự đoán.",
            acc_knn=accuracy_results.get('knn'),
            acc_id3=accuracy_results.get('id3'),
            acc_nb=accuracy_results.get('nb')
        )

    cleaned = clean_text(input_text)
    vec = vectorizer.transform([cleaned])

    predictions = {}
    for name, model in trained_models.items():
        pred = model.predict(vec)
        label = label_encoder.inverse_transform(pred)[0]
        if name == "KNN":
            predictions['pred_knn'] = label
        elif name == "Decision Tree":
            predictions['pred_id3'] = label
        else:
            predictions['pred_nb'] = label

    return render_template(
        'index.html',
        acc_knn=accuracy_results.get('knn'),
        acc_id3=accuracy_results.get('id3'),
        acc_nb=accuracy_results.get('nb'),
        **predictions
    )

if __name__ == '__main__':
    app.run(debug=True)