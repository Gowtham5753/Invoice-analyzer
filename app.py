# app.py — InvoiceFlow: AI Invoice Analyzer with Q&A
import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import google.generativeai as genai

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.environ.get('SECRET_KEY', 'supersecretkey')

# Allow all origins (Netlify proxy will handle this, but keep open for safety)
CORS(app)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Gemini setup — read from environment variable
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDQIEbdAo2Vzc9OS7J8CRU51oBSVlOzJNg")
genai.configure(api_key=GEMINI_API_KEY)

# In-memory store for the latest invoice (single-user demo)
_invoice_store = {
    "encoded_file": None,
    "mime_type": None,
    "summary": None,
    "filename": None,
}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_invoice_with_gemini(encoded_file, mime_type):
    """Analyze an invoice and return a friendly summary."""
    prompt = """
    You are a smart, friendly invoice assistant. Analyze the uploaded invoice and provide a clear, well-structured summary.

    Format your response with these sections using markdown:
    
    ## 📋 Invoice Overview
    - **From:** (company/sender name)
    - **Invoice #:** (number)
    - **Date:** (invoice date)
    - **Due Date:** (if available)
    
    ## 💰 Financial Details
    - **Subtotal:** (amount)
    - **Tax:** (if applicable)
    - **Total Due:** (total amount — make this stand out)
    
    ## 📝 Line Items
    List each item/service with its amount.
    
    ## 📌 Key Notes
    Any payment terms, late fees, or important details worth noting.

    Keep the tone professional but approachable. Use emojis sparingly for visual appeal.
    If any field is not found, simply skip it — don't mention it's missing.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content([
        {
            "inline_data": {
                "mime_type": mime_type,
                "data": encoded_file
            }
        },
        {"text": prompt}
    ])
    return response.text.strip()


def ask_about_invoice(encoded_file, mime_type, question, chat_history=""):
    """Answer a question about the uploaded invoice."""
    prompt = f"""
    You are a helpful invoice assistant. The user has uploaded an invoice and wants to ask questions about it.
    
    Previous conversation context:
    {chat_history}
    
    User's question: {question}
    
    Answer the question based on the invoice. Be concise, helpful, and friendly.
    If the question is unrelated to the invoice, politely redirect them.
    Use markdown formatting for clarity. Keep answers focused and under 200 words unless more detail is needed.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    response = model.generate_content([
        {
            "inline_data": {
                "mime_type": mime_type,
                "data": encoded_file
            }
        },
        {"text": prompt}
    ])
    return response.text.strip()


@app.route('/')
def index():
    return jsonify({"status": "InvoiceFlow API is running"})


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'invoice' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['invoice']
    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            with open(filepath, 'rb') as f:
                file_data = f.read()
            encoded_file = base64.b64encode(file_data).decode('utf-8')

            ext = filename.lower().rsplit('.', 1)[-1]
            if ext == 'pdf':
                mime_type = 'application/pdf'
            elif ext == 'png':
                mime_type = 'image/png'
            else:
                mime_type = 'image/jpeg'

            summary = analyze_invoice_with_gemini(encoded_file, mime_type)

            # Store for Q&A
            _invoice_store["encoded_file"] = encoded_file
            _invoice_store["mime_type"] = mime_type
            _invoice_store["summary"] = summary
            _invoice_store["filename"] = filename

            return jsonify({"summary": summary, "filename": filename})

        except Exception as e:
            return jsonify({"error": f"Error processing invoice: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file type. Upload PNG, JPG, JPEG, or PDF."}), 400


@app.route('/api/ask', methods=['POST'])
def ask():
    """API endpoint for Q&A about the uploaded invoice."""
    data = request.get_json()
    question = data.get('question', '').strip()
    history = data.get('history', '')

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    if not _invoice_store["encoded_file"]:
        return jsonify({"error": "No invoice uploaded. Please upload an invoice first."}), 400

    try:
        answer = ask_about_invoice(
            _invoice_store["encoded_file"],
            _invoice_store["mime_type"],
            question,
            history
        )
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"AI error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
