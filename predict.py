from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import pandas as pd
from datetime import datetime

# Karakteristik daun yang lebih detail
leaf_characteristics = {
    'bambujapan': {
        'tekstur': 'halus',
        'urat': 'sejajar, tidak menonjol',
        'kandungan_air': 'sedang',
        'detail': 'Tekstur halus, urat daun sejajar, tidak cocok untuk ecoprint'
    },
    'binahong': {
        'tekstur': 'halus',
        'urat': 'melengkung, tidak menonjol',
        'kandungan_air': 'tinggi',
        'detail': 'Tekstur halus, bentuk hati, kandungan air tinggi, tidak cocok untuk ecoprint'
    },
    'bodhi': {
        'tekstur': 'kasar',
        'urat': 'menonjol',
        'kandungan_air': 'rendah',
        'detail': 'Tekstur kasar, urat menonjol, cocok untuk ecoprint'
    },
    'jarakmerah': {
        'tekstur': 'kasar',
        'urat': 'menonjol',
        'kandungan_air': 'rendah',
        'detail': 'Tekstur kasar, tepi bergerigi, urat menonjol, cocok untuk ecoprint'
    },
    'jati': {
        'tekstur': 'sangat kasar',
        'urat': 'sangat menonjol',
        'kandungan_air': 'rendah',
        'detail': 'Tekstur sangat kasar, urat sangat menonjol, sangat cocok untuk ecoprint'
    },
    'kamboja': {
        'tekstur': 'halus',
        'urat': 'tidak menonjol',
        'kandungan_air': 'tinggi',
        'detail': 'Tekstur halus, tepi rata, tidak cocok untuk ecoprint'
    },
    'kayuafrika': {
        'tekstur': 'kasar',
        'urat': 'menonjol',
        'kandungan_air': 'rendah',
        'detail': 'Tekstur kasar, urat menonjol, cocok untuk ecoprint'
    },
    'lanang': {
        'tekstur': 'kasar',
        'urat': 'menonjol',
        'kandungan_air': 'rendah',
        'detail': 'Tekstur kasar, bentuk lonjong, urat menonjol, cocok untuk ecoprint'
    },
    'palemjamrud': {
        'tekstur': 'halus',
        'urat': 'tidak menonjol',
        'kandungan_air': 'sedang',
        'detail': 'Tekstur halus, bentuk kipas, tidak cocok untuk ecoprint'
    },
    'tulak': {
        'tekstur': 'halus',
        'urat': 'tidak menonjol',
        'kandungan_air': 'tinggi',
        'detail': 'Tekstur halus, mengkilap, tidak cocok untuk ecoprint'
    }
}

IMG_SIZE = 224

def is_suitable_for_ecoprint(characteristics):
    """Menentukan apakah daun cocok untuk ecoprint berdasarkan karakteristiknya"""
    return (
        characteristics['tekstur'] in ['kasar', 'sangat kasar'] and
        'menonjol' in characteristics['urat'] and
        characteristics['kandungan_air'] == 'rendah'
    )

def save_predictions_to_csv(predictions, filename='prediction_results.csv'):
    """Menyimpan hasil prediksi ke CSV"""
    results_list = []
    
    for img_name, pred in predictions.items():
        result_dict = {
            'timestamp': datetime.now(),
            'image_name': img_name,
            'predicted_class': pred['class'],
            'confidence': pred['confidence'],
            'tekstur': pred['tekstur'],
            'urat': pred['urat'],
            'kandungan_air': pred['kandungan_air'],
            'suitable_for_ecoprint': pred['suitable_for_ecoprint'],
            'is_confident': pred['is_confident']
        }
        
        # Tambahkan top 3 predictions
        for i, (class_name, prob) in enumerate(pred['all_probabilities']):
            result_dict[f'top_{i+1}_class'] = class_name
            result_dict[f'top_{i+1}_probability'] = prob
        
        results_list.append(result_dict)
    
    # Buat DataFrame dan simpan ke CSV
    df = pd.DataFrame(results_list)
    df.to_csv(filename, index=False)
    print(f"\nHasil prediksi telah disimpan ke {filename}")

def predict_image(model, img_path):
    try:
        # Load dan preprocess gambar
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.resnet_v2.preprocess_input(x)
        
        # Prediksi dengan batch prediction
        predictions = model.predict(x, verbose=0)
        
        # Dapatkan nama kelas
        class_names = list(leaf_characteristics.keys())
        
        # Hitung probabilitas
        class_probabilities = {
            name: float(prob) 
            for name, prob in zip(class_names, predictions[0])
        }
        
        # Sort predictions
        sorted_predictions = sorted(
            class_probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Get top prediction
        predicted_class = sorted_predictions[0][0]
        confidence = sorted_predictions[0][1]
        
        # Get characteristics
        char = leaf_characteristics[predicted_class]
        
        return {
            'class': predicted_class,
            'characteristics': char['detail'],
            'tekstur': char['tekstur'],
            'urat': char['urat'],
            'kandungan_air': char['kandungan_air'],
            'confidence': confidence,
            'suitable_for_ecoprint': is_suitable_for_ecoprint(char),
            'all_probabilities': sorted_predictions[:3],
            'is_confident': confidence > 0.7
        }
    
    except Exception as e:
        print(f"Error detail: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        model = load_model('model_daun_ecoprint_resnet.h5')
        
        test_directory = './test_images'
        if os.path.exists(test_directory):
            predictions = {}  # Untuk menyimpan semua hasil prediksi
            
            for img_name in os.listdir(test_directory):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(test_directory, img_name)
                    result = predict_image(model, img_path)
                    
                    if result:
                        predictions[img_name] = result  # Simpan hasil prediksi
                        
                        # Print hasil seperti sebelumnya
                        print(f"\nHasil prediksi untuk {img_name}:")
                        print(f"Kelas: {result['class']}")
                        print(f"Karakteristik Detail:")
                        print(f"- Tekstur: {result['tekstur']}")
                        print(f"- Urat: {result['urat']}")
                        print(f"- Kandungan Air: {result['kandungan_air']}")
                        print(f"Confidence: {result['confidence']:.2%}")
                        print(f"Cocok untuk ecoprint: {result['suitable_for_ecoprint']}")
                        print("\nTop 3 prediksi:")
                        for class_name, prob in result['all_probabilities']:
                            char = leaf_characteristics[class_name]
                            print(f"- {class_name}: {prob:.2%}")
                            print(f"  {char['detail']}")
                        print(f"\nPrediksi dapat dipercaya: {result['is_confident']}")
            
            # Simpan semua hasil ke CSV
            if predictions:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f'prediction_results_resnet_{timestamp}.csv'
                save_predictions_to_csv(predictions, csv_filename)
                
                # Simpan statistik agregat
                aggregate_stats = {
                    'timestamp': [datetime.now()],
                    'total_images': [len(predictions)],
                    'suitable_for_ecoprint': [sum(1 for p in predictions.values() if p['suitable_for_ecoprint'])],
                    'confident_predictions': [sum(1 for p in predictions.values() if p['is_confident'])],
                    'avg_confidence': [np.mean([p['confidence'] for p in predictions.values()])]
                }
                
                # Hitung distribusi kelas
                class_distribution = {}
                for pred in predictions.values():
                    class_name = pred['class']
                    class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                
                for class_name, count in class_distribution.items():
                    aggregate_stats[f'count_{class_name}'] = [count]
                
                # Simpan statistik ke CSV terpisah
                stats_df = pd.DataFrame(aggregate_stats)
                stats_filename = f'prediction_stats_resnet_{timestamp}.csv'
                stats_df.to_csv(stats_filename, index=False)
                print(f"Statistik prediksi telah disimpan ke {stats_filename}")
                
        else:
            print("Folder test_images tidak ditemukan")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        # Simpan error log
        error_log = {
            'timestamp': [datetime.now()],
            'error_message': [str(e)],
            'model_type': ['ResNet']
        }
        pd.DataFrame(error_log).to_csv('prediction_error_log_resnet.csv', index=False)