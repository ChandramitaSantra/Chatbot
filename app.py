from flask import Flask, request, jsonify, render_template, Response, redirect
from transformers import AutoTokenizer, AutoModel
import fitz  # PyMuPDF
import uuid
import chromadb
from collections import defaultdict

app = Flask(__name__)

# Initialize Chroma client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="documents")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# In-memory storage for chat sessions and history
chat_sessions = {}
chat_history = defaultdict(list)

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    return text

def generate_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings.flatten().tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and file.filename.endswith('.txt'):
            text = file.read().decode('utf-8').strip()
            asset_id = str(uuid.uuid4())
            collection.add(documents=[text], ids=[asset_id])
            return jsonify({'asset_id': asset_id})

        elif file and file.filename.endswith('.pdf'):
            text = extract_text_from_pdf(file)
            asset_id = str(uuid.uuid4())
            collection.add(documents=[text], ids=[asset_id])
            return jsonify({'asset_id': asset_id})
        
    return render_template('index.html')

@app.route('/api/documents/process', methods=['POST'])
def process_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith('.txt'):
        text = file.read().decode('utf-8').strip()
    else:
        return jsonify({'error': 'Unsupported file type'}), 400

    embeddings = generate_embeddings(text)
    asset_id = str(uuid.uuid4())
    collection.add(embeddings=[embeddings], ids=[asset_id])

    return jsonify({'asset_id': asset_id}), 200

@app.route('/api/chat/start', methods=['POST'])
def start_chat():
    data = request.json
    asset_id = data.get('asset_id')

    if not asset_id:
        return jsonify({'error': 'Asset ID is required'}), 400

    chat_id = str(uuid.uuid4())
    chat_sessions[chat_id] = asset_id
    return jsonify({'chat_id': chat_id}), 200

@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    data = request.json
    chat_id = data.get('chat_id')
    user_message = data.get('message')

    if not chat_id or not user_message:
        return jsonify({'error': 'Chat ID and message are required'}), 400

    if chat_id not in chat_sessions:
        return jsonify({'error': 'Invalid Chat ID'}), 404

    asset_id = chat_sessions[chat_id]
    response_message = f"ECHO: {user_message}"
    chat_history[chat_id].append({'user': user_message, 'bot': response_message})

    return jsonify({'response': response_message}), 200

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    def generate_response():
        data = request.json
        chat_id = data.get('chat_id')
        user_message = data.get('message')

        if not chat_id or not user_message:
            yield 'data: {"error": "Chat ID and message are required"}\n\n'
            return

        if chat_id not in chat_sessions:
            yield 'data: {"error": "Invalid Chat ID"}\n\n'
            return

        asset_id = chat_sessions[chat_id]
        response_message = f"ECHO: {user_message}"
        chat_history[chat_id].append({'user': user_message, 'bot': response_message})

        for chunk in response_message:
            yield f"data: {chunk}\n\n"
    
    return Response(stream_with_context(generate_response()), mimetype="text/event-stream")

@app.route('/api/chat/history', methods=['GET'])
def chat_history_endpoint():
    chat_id = request.args.get('chat_id')

    if not chat_id:
        return jsonify({'error': 'Chat ID is required'}), 400

    if chat_id not in chat_history:
        return jsonify({'error': 'Invalid Chat ID'}), 404

    return jsonify({'history': chat_history[chat_id]}), 200

if __name__ == '__main__':
    app.run(debug=True)
