import os
import google.generativeai as generai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import PIL.Image as Image
import io
import base64

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

app = Flask(__name__)

# --- Configuração da API Key Gemini ---
API_KEY_GEMINI = os.getenv("API_KEY_GEMINI")

if not API_KEY_GEMINI:
    raise ValueError("A chave API do Gemini não foi encontrada. Defina a variável de ambiente 'API_KEY_GEMINI'.")

generai.configure(api_key=API_KEY_GEMINI)

# Modelos Gemini
# Configuração para controle da geração de respostas (temperatura baixa para mais precisão)
generation_config = {
    "temperature": 0.2,  # Tente um valor baixo para respostas mais factuais
    "top_p": 0.9,        # Parâmetro de amostragem
    "top_k": 40,         # Parâmetro de amostragem
    "max_output_tokens": 800, # Limite o tamanho da resposta em tokens
}

# Modelos Gemini
model_text = generai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)
model_vision = generai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config
)

# Dicionário para armazenar o histórico de conversas por sessão (simples)
chat_sessions = {}

# --- Rotas do Flask ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    session_id = request.headers.get('X-Session-ID', 'default_session') 

    if not user_message and not ( 'image' in data and data['image'] ):
        return jsonify({"response": "Por favor, digite uma mensagem ou envie uma imagem."}), 400

    if session_id not in chat_sessions:
        chat_sessions[session_id] = model_text.start_chat(history=[])

    chat_instance = chat_sessions[session_id]
    gemini_response = ""

    try:
        if 'image' in data and data['image']:
            image_data = data['image'].split(',')[1]
            img_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(img_bytes))

            if not user_message.strip():
                user_message = "Descreva esta imagem."

            response_parts = [user_message, img]
            response = model_vision.generate_content(response_parts)
            gemini_response = response.text

        else:
            response = chat_instance.send_message(user_message)
            gemini_response = response.text

        return jsonify({"response": gemini_response})

    except Exception as e:
        print(f"Erro no Gemini: {e}")
        return jsonify({"response": f"Ops! Ocorreu um erro ao conversar com o Gemini. Tente novamente mais tarde. Detalhes: {str(e)}."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)