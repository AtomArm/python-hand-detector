import cv2
import os

# Configurações de Caminho (Ajuste conforme sua pasta)
PROJECT_PATH = "HaarCascade\HandBatchDetectorHaarCascade.py"
input_image_path = os.path.join(PROJECT_PATH, "images", "input.jpg")
cascade_path = os.path.join(PROJECT_PATH, "resources", "hand.xml")
output_dir = os.path.join(PROJECT_PATH, "images", "output")
os.makedirs(output_dir, exist_ok=True)

def main():
    # 1. Carregar Imagem
    src = cv2.imread(input_image_path)
    if src is None:
        print(f"❌ Erro: Não foi possível carregar a imagem em {input_image_path}")
        return

    # 2. Carregar Classificador
    if not os.path.exists(cascade_path):
        print(f"❌ Erro: Arquivo XML não encontrado em {cascade_path}")
        return
    
    hand_detector = cv2.CascadeClassifier(cascade_path)

    # 3. Detecção (Mesmos parâmetros do Java)
    # scaleFactor=1.1, minNeighbors=1, minSize=(150, 150)
    hands = hand_detector.detectMultiScale(
        src,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(150, 150),
        flags=0
    )

    # 4. Desenhar Retângulos
    for (x, y, w, h) in hands:
        cv2.rectangle(src, (x, y), (x+w, y+h), (0, 255, 0), 10)

    # 5. Salvar Resultado
    output_file = os.path.join(output_dir, "hands_python.png")
    cv2.imwrite(output_file, src)
    
    print(f"✅ Python: Detectado {len(hands)} mãos.")
    print(f"Salvo em: {output_file}")

if __name__ == "__main__":
    main()