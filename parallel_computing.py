import os
import time
import joblib
import pandas as pd
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import models, layers
import tensorflow as tf

def save_semester_data(X_train, X_test, y_train, y_test, semester, data_folder):
    sem_folder = os.path.join(data_folder, f'sem_{semester}')
    os.makedirs(sem_folder, exist_ok=True)
    
    # Simpan data dalam format numpy
    np.save(os.path.join(sem_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(sem_folder, 'X_test.npy'), X_test)
    np.save(os.path.join(sem_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(sem_folder, 'y_test.npy'), y_test)
    
    print(f"Data semester {semester} disimpan di: {sem_folder}")

def save_prediction_results(X_test, y_true, y_pred, semester, results_folder):
    sem_folder = os.path.join(results_folder, f'sem_{semester}')
    os.makedirs(sem_folder, exist_ok=True)
    
    results_df = pd.DataFrame(X_test)
    results_df['true_label'] = np.argmax(y_true, axis=1)
    results_df['predicted_label'] = np.argmax(y_pred, axis=1)
    results_df['prediction_correct'] = results_df['true_label'] == results_df['predicted_label']
    
    csv_path = os.path.join(sem_folder, 'predictions.csv')
    results_df.to_csv(csv_path, index=False)
    
    np.save(os.path.join(sem_folder, 'y_true.npy'), y_true)
    np.save(os.path.join(sem_folder, 'y_pred.npy'), y_pred)
    
    print(f"Hasil prediksi semester {semester} disimpan di: {csv_path}")

def evaluate_with_custom_metrics(model, X_test, y_test, semester, results_folder):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)
    # Simpan hasil prediksi
    save_prediction_results(X_test, y_test, y_pred_probs, semester, results_folder)

    num_classes = y_test.shape[1]
    precisions = []
    recalls = []
    f1s = []

    for i in range(num_classes):
        y_true_i = (y_true == i).astype(int)
        y_pred_i = (y_pred == i).astype(int)

        precision = tf.keras.metrics.Precision()
        recall = tf.keras.metrics.Recall()

        precision.update_state(y_true_i, y_pred_i)
        recall.update_state(y_true_i, y_pred_i)

        p = precision.result().numpy()
        r = recall.result().numpy()

        f1 = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=25,
    restore_best_weights=True,
    mode='max',
    verbose=1
)

def build_dnn_model(input_shape, output_units):
    model = models.Sequential()
    
    # Input layer
    model.add(layers.Dense(256, activation='relu', input_shape=(input_shape,)))
    
    # Hidden layers
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(output_units, activation='softmax'))
    # optimizer = Adam(learning_rate=0.0005)

    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model


dropped_column = ['target', 'is_disabilitas_mhs','sks_status_sem',
                    'status_nikah_mhs', 'nama_negara_mhs', 'pekerjaan_mhs', 'jenis_kelamin_mhs',
                    'jumlah_dosen_pembimbing_prodi', 'sks_lulus_min_prodi', 'gelombang_mhs', 'is_krs_diajukan_sem', 
                    'is_paket_sem', 'is_krs_disetujui_sem', 'is_krs_disetujui_sem', 'batas_sks_sem', 'sks_semester_sem',
                    'agama_mhs', 'nilai_sks_total_sem', 'penghasilan_ibu_mhs', 'ipk_asal_mhs', 'sks_asal_mhs',
                    'lembaga_akreditasi_prodi', 'jumlah_dosen_penguji_prodi', 'jenis_tugas_akhir_prodi', 'status_mahasiswa_sem', 'is_transfer_mhs',
                    'sks_total_sem', 'semester_mata_kuliah_sem', 'sks_lulus_sem',
                    'penghasilan_ayah_mhs', 'nilai_sks_semester_sem', 'nilai_sks_lulus_sem', 'ipk_lulus_sem'
                ]

def process_semester(sem, input_folder, output_base_folder):
    print(f'\n==== Processing semester {sem} ====')
    try:
        sem_output_folder = os.path.join(output_base_folder, f'sem_{sem}')
        os.makedirs(sem_output_folder, exist_ok=True)
        
        df = pd.read_csv(os.path.join(input_folder, f'sem_{sem}.csv'))
        X = df.drop(columns=dropped_column)
        y = df['target']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        encoder = OneHotEncoder(sparse_output=False)
        y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42
        )
        
        save_semester_data(X_train, X_test, y_train, y_test, sem, 
                          os.path.join(output_base_folder, 'split_data'))

        joblib.dump(scaler, os.path.join(sem_output_folder, 'scaler.pkl'))
        joblib.dump(encoder, os.path.join(sem_output_folder, 'encoder.pkl'))

        model = build_dnn_model(X_train.shape[1], y_encoded.shape[1])
        
        model_path = os.path.join(sem_output_folder, 'model.keras')
        checkpoint = ModelCheckpoint(
            filepath=model_path,
            save_weights_only=False,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=0
        )

        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
        training_time = round(time.time() - start_time, 2)

        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        precision, recall, f1 = evaluate_with_custom_metrics(
            model, X_test, y_test, sem, 
            os.path.join(output_base_folder, 'predictions')
        )

        pd.DataFrame(history.history).to_csv(
            os.path.join(sem_output_folder, 'training_history.csv'), 
            index=False
        )

        return {
            'semester': sem,
            'loss': round(loss, 4),
            'accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'training_time': training_time,
            'model_path': model_path
        }

    except Exception as e:
        print(f"Error processing semester {sem}: {str(e)}")
        return None

def main():
    try:
        base_dir = 'D:/riset-kelulusan'
        input_folder = os.path.join(base_dir, 'data_persemester/data_kedinasan_smote')
        output_base_folder = os.path.join(base_dir, 'results/processed_semesters')
        
        os.makedirs(output_base_folder, exist_ok=True)
        os.makedirs(os.path.join(output_base_folder, 'split_data'), exist_ok=True)
        os.makedirs(os.path.join(output_base_folder, 'predictions'), exist_ok=True)

        df = pd.read_csv(os.path.join(base_dir, 'data_train/df_smote_kedinasan.csv'), index_col=0)
        semesters = df['semester_mahasiswa_sem'].unique()
        print(f"Found {len(semesters)} semesters to process")

        start_time = time.time()
        total_start_time = time.time()  

        # with ThreadPoolExecutor(max_workers=(14)) as executor:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_semester, sem, input_folder, output_base_folder) 
                      for sem in semesters]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"Completed semester {result['semester']} with accuracy {result['accuracy']:.4f}")
                except Exception as e:
                    print(f"Error in future: {e}")
        
        total_end_time = time.time()
        total_wall_time = round(total_end_time - total_start_time, 2)
        if results:
            summary_df = pd.DataFrame(results)

            summary_path = os.path.join(output_base_folder, 'summary_results.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary saved to {summary_path}")
            
            total_time = round(time.time() - start_time, 2)
            avg_accuracy = summary_df['accuracy'].mean()
            print(f"\nProcessing completed in {total_time} seconds")
            print(f"\nTotal wall time: {total_wall_time} seconds") 
            print(f"Average accuracy across all semesters: {avg_accuracy:.4f}")
            
            top3 = summary_df.nlargest(3, 'accuracy')
            print("\nTop 3 semesters by accuracy:")
            print(top3[['semester', 'accuracy']].to_string(index=False))
            
        else:
            print("No semesters were processed successfully")

    except Exception as e:
        print(f"Main execution error: {str(e)}")

if __name__ == '__main__':
    main()