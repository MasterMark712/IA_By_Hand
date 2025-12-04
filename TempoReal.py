import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyautogui  # Importa a biblioteca para pressionar teclas
import pydirectinput
from sklearn.preprocessing import LabelEncoder

# Carrega o modelo treinado
model = tf.keras.models.load_model('modelo_gestos.h5')

# Defina as classes na mesma ordem usada durante o treinamento
classes = ['PULA', 'FRENTE', 'VERSO', 'NADA']

# Recria o LabelEncoder com as classes
le = LabelEncoder()
le.fit(classes)

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Controle do estado das teclas
key_states = {'PULA': False, 'FRENTE': False, 'VERSO': False}

print("Pressione 'ESC' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inverte a imagem horizontalmente para efeito de espelho
    frame = cv2.flip(frame, 1)

    # Converte a imagem para RGB (necessário para o MediaPipe)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Processa a imagem e extrai os keypoints
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Desenha os keypoints e conexões na imagem
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extrai as coordenadas normalizadas
        keypoints = []
        for lm in hand_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])

        keypoints = np.array(keypoints).flatten()

        # Normaliza os keypoints em relação ao pulso (ponto 0)
        wrist = keypoints[:3]
        keypoints_normalized = keypoints - np.tile(wrist, 21)

        # Redimensiona para o formato de entrada do modelo
        keypoints_normalized = keypoints_normalized.reshape(1, -1)

        # Prediz a classe do gesto
        prediction = model.predict(keypoints_normalized)
        class_id = np.argmax(prediction)
        class_name = le.inverse_transform([class_id])[0]

        # Exibe a classe na imagem
        cv2.putText(frame, class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 3, cv2.LINE_AA)

        # Lógica para pressionar ou soltar a tecla dependendo do estado do gesto
        if class_name == 'PULA' and not key_states['PULA']:
            pydirectinput.keyDown('s')  # Pressiona a tecla 's'
            key_states['PULA'] = True
        elif class_name != 'PULA' and key_states['PULA']:
            pydirectinput.keyUp('s')  # Solta a tecla 's'
            key_states['PULA'] = False

        elif class_name == 'FRENTE' and not key_states['FRENTE']:
            pydirectinput.keyDown('right')  # Pressiona a seta para a direita
            key_states['FRENTE'] = True
        elif class_name != 'FRENTE' and key_states['FRENTE']:
            pydirectinput.keyUp('right')  # Solta a seta para a direita
            key_states['FRENTE'] = False

        elif class_name == 'VERSO' and not key_states['VERSO']:
            pydirectinput.keyDown('left')  # Pressiona a seta para a esquerda
            key_states['VERSO'] = True
        elif class_name != 'VERSO' and key_states['VERSO']:
            pydirectinput.keyUp('left')  # Solta a seta para a esquerda
            key_states['VERSO'] = False

        # Se for 'NADA', não faz nenhuma ação, mantendo as teclas soltas

    else:
        # Se nenhuma mão for detectada, exibe mensagem
        cv2.putText(frame, 'Posicione-se para Iniciar!', (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # Exibe o frame
    cv2.imshow('Reconhecimento de Gestos', frame)

    # Sai com a tecla ESC
    if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC
        break

# Libera os recursos
cap.release()
cv2.destroyAllWindows()
