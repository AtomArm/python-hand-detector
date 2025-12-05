import cv2
import os
import csv
import time
import numpy as np

# --- CONFIGURAÇÃO DE CAMINHOS ---
# Ajuste o BASE_PATH para onde está sua pasta 'benchmark'
BASE_PATH = "benchmark/" 

XML_PATH = os.path.join(BASE_PATH, "resources", "hand.xml")
INPUT_DIR = os.path.join(BASE_PATH, "images", "images")
OUTPUT_CSV = os.path.join(BASE_PATH, "csvs", "batch_results_haar_python.csv")
OUTPUT_IMG_DIR = os.path.join(BASE_PATH, "images", "processed_haar_python")

def main():
    # 1. Verificar diretórios
    if not os.path.exists(XML_PATH):
        print(f"❌ Erro: XML não encontrado em {XML_PATH}")
        return
    
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # 2. Carregar Classificador
    hand_detector = cv2.CascadeClassifier(XML_PATH)
    if hand_detector.empty():
        print("❌ Erro ao carregar o classificador Cascade.")
        return

    print("🚀 Iniciando Batch Processing (Python + Haar)...")

    # 3. Preparar CSV
    with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
        fieldnames = ['file', 'detected_count', 'time_ns', 'first_x', 'first_y', 'first_w', 'first_h']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # 4. Iterar sobre as imagens
        files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for filename in files:
            image_path = os.path.join(INPUT_DIR, filename)
            frame = cv2.imread(image_path)
            
            if frame is None:
                print(f"⚠️ Erro ao ler imagem: {filename}")
                continue

            # --- INÍCIO DA MEDIÇÃO DE TEMPO ---
            # perf_counter_ns é o relógio mais preciso para benchmarking em Python
            start_time = time.perf_counter_ns()

            # Detecção com PARÂMETROS RIGOROSOS (Iguais ao Java)
            # scaleFactor=1.1, minNeighbors=1, minSize=(150, 150)
            hands = hand_detector.detectMultiScale(
                frame,
                scaleFactor=1.1,
                minNeighbors=1,
                minSize=(150, 150),
                flags=0
            )

            end_time = time.perf_counter_ns()
            duration_ns = end_time - start_time
            # --- FIM DA MEDIÇÃO ---

            count = len(hands)
            
            # Dados para CSV e Desenho
            first_x, first_y, first_w, first_h = 0, 0, 0, 0
            
            if count > 0:
                # Desenhar retângulos
                for i, (x, y, w, h) in enumerate(hands):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Pegar o primeiro para o CSV
                    if i == 0:
                        first_x, first_y, first_w, first_h = x, y, w, h

            # Escrever na tela (Overlay)
            time_ms = duration_ns / 1_000_000.0
            cv2.putText(frame, f"Hands: {count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Time: {time_ms:.2f} ms", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Salvar imagem
            output_path = os.path.join(OUTPUT_IMG_DIR, filename)
            cv2.imwrite(output_path, frame)

            # Gravar no CSV
            writer.writerow({
                'file': filename,
                'detected_count': count,
                'time_ns': duration_ns,
                'first_x': first_x,
                'first_y': first_y,
                'first_w': first_w,
                'first_h': first_h
            })

            print(f"✅ Processado: {filename} | Mãos: {count} | Tempo: {time_ms:.2f}ms")

    print(f"🏁 Finalizado. CSV salvo em: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()