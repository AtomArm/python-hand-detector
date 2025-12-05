import cv2
import os
import csv
import time
import numpy as np
import random

# --- CONFIGURAÇÃO DE CAMINHOS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "benchmark")

XML_PATH = os.path.join(BASE_PATH, "resources", "hand.xml")

INPUT_DIR = os.path.join(BASE_PATH, "images", "dataset") 

OUTPUT_CSV = os.path.join(BASE_PATH, "csvs", "batch_results_haar_python" + str(random.randint(0,999)) + ".csv")
OUTPUT_IMG_DIR = os.path.join(BASE_PATH, "images", "processed_haar_python")

def main():
    print(f"📂 Diretório do Script: {SCRIPT_DIR}")
    print(f"📂 Varrendo subpastas em: {INPUT_DIR}")
    
    if not os.path.exists(XML_PATH):
        print(f"❌ Erro: XML não encontrado em {XML_PATH}")
        return
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Erro: Pasta de imagens não encontrada.")
        return

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    hand_detector = cv2.CascadeClassifier(XML_PATH)
    if hand_detector.empty():
        print("❌ Erro ao carregar XML.")
        return

    print("🚀 Iniciando Batch Processing...")

    with open(OUTPUT_CSV, mode='w', newline='') as csv_file:
        fieldnames = ['file', 'type', 'detected_count', 'time_ns', 'first_x', 'first_y', 'first_w', 'first_h']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        processed_count = 0

        for root, dirs, files in os.walk(INPUT_DIR):
            
            # --- MUDANÇA AQUI: Pega o nome da pasta atual como 'Tipo' ---
            # Se o caminho for .../allimgs/fingerCircle, isso pega "fingerCircle"
            current_folder_name = os.path.basename(root)
            
            # Se estiver na raiz (allimgs), definimos como 'geral' ou o próprio nome da pasta
            if current_folder_name == "allimgs":
                image_type = "root" 
            else:
                image_type = current_folder_name

            for filename in files:
                if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                image_path = os.path.join(root, filename)
                frame = cv2.imread(image_path)
                
                if frame is None:
                    continue

                # --- MEDIÇÃO ---
                start_time = time.perf_counter_ns()

                hands = hand_detector.detectMultiScale(
                    frame,
                    scaleFactor=1.1,
                    minNeighbors=1,
                    minSize=(150, 150),
                    flags=0
                )

                end_time = time.perf_counter_ns()
                duration_ns = end_time - start_time
                # --- FIM MEDIÇÃO ---

                count = len(hands)
                first_x, first_y, first_w, first_h = 0, 0, 0, 0
                
                if count > 0:
                    for i, (x, y, w, h) in enumerate(hands):
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        if i == 0:
                            first_x, first_y, first_w, first_h = x, y, w, h

                # Nome do arquivo de saída (para não misturar se tiver nomes iguais em pastas diferentes)
                unique_filename = f"{image_type}_{filename}"
                output_path = os.path.join(OUTPUT_IMG_DIR, unique_filename)
                
                # Overlay
                time_ms = duration_ns / 1_000_000.0
                cv2.putText(frame, f"Hands: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Type: {image_type}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imwrite(output_path, frame)

                writer.writerow({
                    'file': filename,
                    'type': image_type,    # Aqui entra o nome da pasta
                    'detected_count': count,
                    'time_ns': duration_ns,
                    'first_x': first_x,
                    'first_y': first_y,
                    'first_w': first_w,
                    'first_h': first_h
                })

                processed_count += 1
                print(f"✅ {image_type} | {filename} | {time_ms:.2f}ms")

    print(f"🏁 Finalizado. CSV salvo em: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()