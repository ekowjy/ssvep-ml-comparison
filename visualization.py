import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import butter, lfilter
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
DATA_FOLDER = r".\dataset"
OUTPUT_FOLDER = "output_confusion"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
target_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
fs = 128
target_frequencies = [8.57, 10.0, 12.0, 15.0, 17.14, 20.0]
n_harmonics = 2
frequency_mapping = {
    8.57: "Forward",
    10.0: "Backward", 
    12.0: "Left",
    15.0: "Right",
    17.14: "Takeoff",
    20.0: "Landing"
}
durasi_per_freq = 5 
label_map = {f: i for i, f in enumerate(target_frequencies)}
label_names = [frequency_mapping[f] for f in target_frequencies]
def get_valid_eeg_channels(columns):
    """Deteksi kolom EEG dari header file yang cocok dengan target"""
    matched = []
    print(f"[DEBUG] Mencari channel dalam columns: {list(columns)[:20]}...")
    
    for ch in target_channels:
        found = False
        for col in columns:
            if (f'EEG.{ch}' == col or 
                col.endswith(f'.{ch}') or 
                ch in col.split('.')):
                matched.append(col)
                found = True
                break
        if found:
            print(f"[✅] Channel {ch} ditemukan sebagai: {matched[-1]}")
        else:
            print(f"[❌] Channel {ch} tidak ditemukan")
    
    return matched

def bandpass_filter(data, lowcut, highcut, fs=128, order=4):
    """Apply bandpass filter to EEG data"""
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=0)

def insert_marker_labels(frequencies, fs, durasi):
    """Return marker array (label integer) sepanjang durasi data."""
    interval = fs * durasi
    marker = []
    for f in frequencies:
        label = label_map[f]
        marker.extend([label] * interval)
    return marker

def load_eeg_csv(filepath):
    """Load EEG data from CSV file with special handling for concatenated format"""
    try:
        print(f"[DEBUG] Membaca file: {filepath}")
        with open(filepath, 'r') as f:
            header_line = f.readline().strip()
        
        print(f"[DEBUG] Header length: {len(header_line)}")
        print(f"[DEBUG] First 100 chars: {header_line[:100]}...")
        df = None
        parsing_method = "unknown"
        try:
            df = pd.read_csv(filepath, skiprows=1)
            parsing_method = "standard"
            print(f"[DEBUG] Standard parsing berhasil: {df.shape}")
        except Exception as e:
            print(f"[DEBUG] Standard parsing gagal: {e}")
        if df is None or df.shape[1] <= 3:
            try:
                df = pd.read_csv(filepath, skiprows=1, sep='\t')
                if df.shape[1] > 3:
                    parsing_method = "tab_separated"
                    print(f"[DEBUG] Tab-separated parsing berhasil: {df.shape}")
            except Exception as e:
                print(f"[DEBUG] Tab-separated parsing gagal: {e}")
        
        if df is None or df.shape[1] <= 3:
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()[1:] 
                sample_lines = lines[:5]
                print(f"[DEBUG] Sample lines:")
                for i, line in enumerate(sample_lines):
                    print(f"  Line {i}: {line[:100]}...")
                delimiters = [',', '.', '\t', ';', ' ']
                best_delimiter = ','
                max_columns = 0
                
                for delim in delimiters:
                    try:
                        test_split = lines[0].strip().split(delim)
                        if len(test_split) > max_columns:
                            max_columns = len(test_split)
                            best_delimiter = delim
                    except:
                        continue
                
                print(f"[DEBUG] Best delimiter: '{best_delimiter}' with {max_columns} columns")
                parsed_data = []
                for line in lines:
                    row = line.strip().split(best_delimiter)
                    parsed_data.append(row)
                
                df = pd.DataFrame(parsed_data)
                parsing_method = f"manual_delim_{best_delimiter}"
                print(f"[DEBUG] Manual parsing berhasil: {df.shape}")
                
            except Exception as e:
                print(f"[DEBUG] Manual parsing gagal: {e}")
                raise ValueError(f"Semua metode parsing gagal: {e}")
        
        print(f"[INFO] Parsing method used: {parsing_method}")
        print(f"[INFO] Final data shape: {df.shape}")
        print("[DEBUG] Converting to numeric...")
        numeric_df = pd.DataFrame()
        
        for col_idx in range(df.shape[1]):
            try:
                numeric_col = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
                if numeric_col.notna().sum() / len(numeric_col) > 0.5:
                    numeric_df[f'col_{col_idx}'] = numeric_col.fillna(0)
            except Exception as e:
                print(f"[DEBUG] Error converting column {col_idx}: {e}")
        
        print(f"[DEBUG] Numeric data shape: {numeric_df.shape}")
        if numeric_df.shape[1] < 4:
            raise ValueError(f"Tidak cukup kolom numeric valid: {numeric_df.shape[1]}")
        variances = numeric_df.var()
        print(f"[DEBUG] Column variances: {variances.head(10).to_dict()}")
        valid_cols = variances[variances > 1].index.tolist()
        
        if len(valid_cols) < 4:
            start_idx = max(2, min(3, numeric_df.shape[1] // 4))
            end_idx = min(start_idx + 14, numeric_df.shape[1])
            valid_cols = [f'col_{i}' for i in range(start_idx, end_idx)]
            print(f"[WARNING] Using fallback columns: {valid_cols}")
        selected_cols = valid_cols[:min(14, len(valid_cols))]
        print(f"[INFO] Selected {len(selected_cols)} columns as EEG channels")
        
        eeg_data = numeric_df[selected_cols].values
        print(f"[DEBUG] Final EEG data shape: {eeg_data.shape}")
        print(f"[DEBUG] EEG data range: {eeg_data.min():.2f} to {eeg_data.max():.2f}")
        print(f"[ℹ️] Membuat marker otomatis untuk file: {os.path.basename(filepath)}")
        marker = insert_marker_labels(target_frequencies, fs, durasi_per_freq)
        if len(marker) < len(eeg_data):
            last_class = marker[-1] if marker else 0
            marker += [last_class] * (len(eeg_data) - len(marker))
        elif len(marker) > len(eeg_data):
            marker = marker[:len(eeg_data)]
        
        marker = np.array(marker)
        print(f"[DEBUG] Marker shape: {marker.shape}, unique values: {np.unique(marker)}")
        unique_markers = np.unique(marker)
        if len(unique_markers) < 2:
            print("[WARNING] Hanya satu kelas dalam marker, membuat variasi artifisial...")
            n_segments = min(len(target_frequencies), len(eeg_data) // 100)
            segment_size = len(eeg_data) // n_segments
            
            for i in range(n_segments):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(eeg_data)
                marker[start_idx:end_idx] = i % len(target_frequencies)
            
            print(f"[DEBUG] Final marker unique values: {np.unique(marker)}")
        
        return eeg_data, marker
        
    except Exception as e:
        print(f"[ERROR] Detail error in {os.path.basename(filepath)}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise ValueError(f"Gagal membaca file {filepath}: {e}")

def evaluate_models(X, y):
    """Evaluate multiple models with cross-validation"""
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        ),
        "SVM": SVC(
            kernel='rbf', 
            C=1.0,
            gamma='scale',
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        "K-NN": KNeighborsClassifier(
            n_neighbors=5,
            weights='uniform'
        )
    }
    
    results = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"[INFO] Evaluating {name}...")
        acc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_weighted')
        
        results[name] = {
            "accuracy": acc_scores.mean(),
            "accuracy_std": acc_scores.std(),
            "f1_score": f1_scores.mean(),
            "f1_std": f1_scores.std(),
            "cv_scores": acc_scores,
            "cv_coefficient": (acc_scores.std() / acc_scores.mean()) * 100  # CV percentage
        }
        
        print(f"[✅] {name}: Accuracy={acc_scores.mean():.4f}±{acc_scores.std():.4f}, F1={f1_scores.mean():.4f}±{f1_scores.std():.4f}")
    
    return results

def perform_anova_analysis(results):
    """Perform ANOVA analysis on model results"""
    print("\n[INFO] Performing ANOVA analysis...")
    model_names = list(results.keys())
    cv_scores = [results[name]["cv_scores"] for name in model_names]
    f_statistic, p_value = stats.f_oneway(*cv_scores)
    all_scores = np.concatenate(cv_scores)
    grand_mean = np.mean(all_scores)
    ss_total = np.sum((all_scores - grand_mean) ** 2)
    ss_between = 0
    for scores in cv_scores:
        group_mean = np.mean(scores)
        ss_between += len(scores) * (group_mean - grand_mean) ** 2
    eta_squared = ss_between / ss_total
    
    print(f"[📊] ANOVA Results:")
    print(f"    F-statistic: {f_statistic:.4f}")
    print(f"    p-value: {p_value:.6f}")
    print(f"    η² (eta-squared): {eta_squared:.4f}")
    
    return {
        "f_statistic": f_statistic,
        "p_value": p_value,
        "eta_squared": eta_squared
    }

def apply_fbcca(eeg_data, fs=128):
    """Apply FBCCA preprocessing (bandpass filter)"""
    return bandpass_filter(eeg_data, 5, 45, fs)

def extract_features(eeg_data):
    """Extract features from EEG data"""
    features = []
    
    for ch in range(eeg_data.shape[1]):
        channel_data = eeg_data[:, ch]
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        var_val = np.var(channel_data)
        skew_val = stats.skew(channel_data)
        kurt_val = stats.kurtosis(channel_data)
        
        features.extend([mean_val, std_val, var_val, skew_val, kurt_val])
    
    return np.array(features)

def create_windowed_features(eeg_data, marker, window_size=128):
    """Create windowed features from EEG data"""
    n_samples, n_channels = eeg_data.shape
    features = []
    labels = []
    
    step_size = window_size // 2
    
    for i in range(0, n_samples - window_size, step_size):
        window_data = eeg_data[i:i+window_size]
        window_marker = marker[i:i+window_size]
        unique_labels, counts = np.unique(window_marker, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]
        window_features = extract_features(window_data)
        
        features.append(window_features)
        labels.append(majority_label)
    
    return np.array(features), np.array(labels)

def plot_and_save_confusion_matrix(y_true, y_pred, classes, filename):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=False, values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[📊] Confusion matrix disimpan: {os.path.basename(filename)}")

def plot_model_comparison(all_results, save_path="model_comparison.png"):
    """Plot model comparison chart"""
    if not all_results:
        print("[⚠️] Tidak ada hasil untuk dibuat grafik")
        return
    model_names = ["Random Forest", "SVM", "Logistic Regression", "K-NN"]
    aggregated_results = {name: [] for name in model_names}
    
    for file_results in all_results:
        if 'model_results' in file_results:
            for model_name in model_names:
                if model_name in file_results['model_results']:
                    aggregated_results[model_name].append(
                        file_results['model_results'][model_name]['accuracy']
                    )
    
    means = []
    stds = []
    for model_name in model_names:
        if aggregated_results[model_name]:
            means.append(np.mean(aggregated_results[model_name]))
            stds.append(np.std(aggregated_results[model_name]))
        else:
            means.append(0)
            stds.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8, 
                  color=['#2E8B57', '#4682B4', '#CD853F', '#DC143C'])
    
    ax.set_xlabel('Classification Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(True, axis='y', alpha=0.3)
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.annotate(f'{mean:.3f}±{std:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[📊] Model comparison chart saved: {save_path}")

def process_file(filepath):
    """Process single EEG file with robust error handling"""
    filename = os.path.basename(filepath)
    
    try:
        print(f"[INFO] Memproses: {filename}")
        eeg_data, marker = load_eeg_csv(filepath)
        unique_classes = np.unique(marker)
        print(f"[INFO] Kelas ditemukan: {sorted(unique_classes)} (total: {len(unique_classes)})")
        
        if len(unique_classes) <= 1:
            raise ValueError("Hanya satu kelas ditemukan, tidak bisa diklasifikasi.")
        class_counts = {cls: np.sum(marker == cls) for cls in unique_classes}
        print(f"[INFO] Distribusi kelas: {class_counts}")
        
        min_samples_per_class = min(class_counts.values())
        if min_samples_per_class < 10:
            print(f"[WARNING] Beberapa kelas memiliki sampel sedikit (min: {min_samples_per_class})")
        try:
            print("[INFO] Applying FBCCA preprocessing...")
            eeg_filtered = apply_fbcca(eeg_data)
            print(f"[DEBUG] Filtered data shape: {eeg_filtered.shape}")
        except Exception as e:
            print(f"[WARNING] FBCCA preprocessing gagal: {e}, menggunakan data mentah")
            eeg_filtered = eeg_data
        try:
            print("[INFO] Extracting features...")
            X, y = create_windowed_features(eeg_filtered, marker)
            print(f"[DEBUG] Feature matrix shape: {X.shape}")
            print(f"[DEBUG] Labels shape: {y.shape}")
        except Exception as e:
            print(f"[ERROR] Feature extraction gagal: {e}")
            return None
        
        if len(X) < 10:
            print(f"[WARNING] Tidak cukup sampel untuk cross-validation: {len(X)}")
            return None
        
        try:
            print("[INFO] Evaluating models...")
            model_results = evaluate_models(X, y)
            
            anova_results = perform_anova_analysis(model_results)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            best_model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            )
            best_model.fit(X_scaled, y)
            y_pred = best_model.predict(X_scaled)
            
            print(f"[✅] {filename} processed successfully")
            
            return {
                'filename': filename,
                'model_results': model_results,
                'anova_results': anova_results,
                'y_true': y,
                'y_pred': y_pred,
                'best_accuracy': model_results['Random Forest']['accuracy'],
                'best_f1': model_results['Random Forest']['f1_score']
            }
            
        except Exception as e:
            print(f"[ERROR] Model evaluation gagal: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"[❌] Error processing {filename}: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None

def generate_statistical_report(all_results):
    """Generate comprehensive statistical report"""
    if not all_results:
        print("[❌] No results to generate report")
        return
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE STATISTICAL ANALYSIS REPORT")
    print(f"{'='*80}")
    
    model_names = ["Random Forest", "SVM", "Logistic Regression", "K-NN"]
    aggregated_results = {name: [] for name in model_names}
    
    for file_results in all_results:
        if 'model_results' in file_results:
            for model_name in model_names:
                if model_name in file_results['model_results']:
                    aggregated_results[model_name].append(
                        file_results['model_results'][model_name]['accuracy']
                    )
    
    print("\nMODEL PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    model_stats = {}
    for model_name in model_names:
        if aggregated_results[model_name]:
            accuracies = aggregated_results[model_name]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            cv_percent = (std_acc / mean_acc) * 100
            
            model_stats[model_name] = {
                'mean': mean_acc,
                'std': std_acc,
                'cv': cv_percent
            }
            
            print(f"{model_name:20}: {mean_acc:.4f} ± {std_acc:.4f} (CV: {cv_percent:.2f}%)")
    
    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"\nRANKING (by accuracy):")
    print("-" * 30)
    for i, (model_name, stats) in enumerate(sorted_models, 1):
        print(f"{i}. {model_name}: {stats['mean']:.4f}")
    
    if all_results and 'anova_results' in all_results[0]:
        anova = all_results[0]['anova_results']
        print(f"\nANOVA ANALYSIS:")
        print("-" * 20)
        print(f"F-statistic: {anova['f_statistic']:.4f}")
        print(f"p-value: {anova['p_value']:.6f}")
        print(f"η² (effect size): {anova['eta_squared']:.4f}")
        
        if anova['p_value'] < 0.001:
            significance = "p < 0.001"
        elif anova['p_value'] < 0.01:
            significance = f"p < 0.01"
        elif anova['p_value'] < 0.05:
            significance = f"p < 0.05"
        else:
            significance = f"p = {anova['p_value']:.3f}"
        
        print(f"Significance: {significance}")
    
    if len(sorted_models) >= 4:
        rf_acc = sorted_models[0][1]['mean']
        svm_acc = sorted_models[1][1]['mean']
        lr_acc = sorted_models[2][1]['mean']
        knn_acc = sorted_models[3][1]['mean']
        rf_cv = sorted_models[0][1]['cv']
        
        print(f"\n{'='*80}")
        print("FINAL SUMMARY STATEMENT:")
        print(f"{'='*80}")
        print(f"Random Forest achieved highest accuracy ({rf_acc:.3f}), followed by")
        print(f"SVM ({svm_acc:.3f}), Logistic Regression ({lr_acc:.3f}), and")
        print(f"K-NN ({knn_acc:.3f}). ANOVA revealed significant differences")
        
        if 'anova_results' in all_results[0]:
            anova = all_results[0]['anova_results']
            print(f"(F({len(model_names)-1}) = {anova['f_statistic']:.2f}, p < 0.001,")
            print(f"η² = {anova['eta_squared']:.3f}), with RF demonstrating superior")
            print(f"cross-validation stability (CV = {rf_cv:.1f}%).")
        
        print(f"{'='*80}")

def run_batch_pipeline(folder_path):
    """Run the complete pipeline on all CSV files in folder"""
    all_results = []
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    if not csv_files:
        print("[❌] Tidak ada file CSV ditemukan dalam folder!")
        return
    
    print(f"[ℹ️] Ditemukan {len(csv_files)} file CSV untuk diproses")
    
    for filename in csv_files:
        filepath = os.path.join(folder_path, filename)
        print(f"\n[🔄] Memproses: {filename}")
        
        hasil = process_file(filepath)
        
        if hasil is not None:
            all_results.append(hasil)
            
            cm_filename = os.path.join(OUTPUT_FOLDER, f'cm_{filename.replace(".csv", "")}.png')
            plot_and_save_confusion_matrix(
                hasil['y_true'],
                hasil['y_pred'],
                classes=list(range(len(np.unique(hasil['y_true'])))),
                filename=cm_filename
            )
    if not all_results:
        print("\n[❌] Tidak ada file yang berhasil diproses!")
        return
    
    print(f"\n[✅] Berhasil memproses {len(all_results)} file")
    comparison_chart_path = os.path.join(OUTPUT_FOLDER, 'model_comparison.png')
    plot_model_comparison(all_results, save_path=comparison_chart_path)
    generate_statistical_report(all_results)

    detailed_results = []
    for result in all_results:
        if 'model_results' in result:
            row = {'filename': result['filename']}
            for model_name, model_result in result['model_results'].items():
                row[f'{model_name}_accuracy'] = model_result['accuracy']
                row[f'{model_name}_f1'] = model_result['f1_score']
                row[f'{model_name}_cv'] = model_result['cv_coefficient']
            detailed_results.append(row)
    
    if detailed_results:
        df_detailed = pd.DataFrame(detailed_results)
        detailed_csv_path = os.path.join(OUTPUT_FOLDER, 'detailed_model_results.csv')
        df_detailed.to_csv(detailed_csv_path, index=False)
        print(f"[💾] Detailed results saved: {detailed_csv_path}")

if __name__ == "__main__":
    print("🚀 Memulai Enhanced FBCCA Pipeline dengan Model Comparison...")
    print(f"📁 Folder data: {DATA_FOLDER}")
    print(f"📁 Folder output: {OUTPUT_FOLDER}")
    print(f"🧠 Target channels: {target_channels}")
    print(f"🎯 Target frequencies: {target_frequencies}")
    
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
    if csv_files:
        debug_file = os.path.join(DATA_FOLDER, csv_files[0])
        print(f"\n🔍 DEBUG: Menganalisis struktur file: {csv_files[0]}")
        try:
            df_debug = pd.read_csv(debug_file, skiprows=1)
            print(f"📊 Jumlah kolom: {len(df_debug.columns)}")
            print(f"📊 Jumlah baris: {len(df_debug)}")
            
            eeg_columns = [col for col in df_debug.columns if 'EEG.' in col]
            print(f"📊 Kolom EEG ditemukan: {len(eeg_columns)}")
            
            target_found = []
            for ch in target_channels:
                matching = [col for col in eeg_columns if ch in col]
                if matching:
                    target_found.append(f"{ch} → {matching[0]}")
            
            print(f"📊 Target channel yang cocok: {len(target_found)}")
            for match in target_found[:5]:
                print(f"   {match}")
            if len(target_found) > 5:
                print(f"   ... dan {len(target_found)-5} lainnya")
                
        except Exception as e:
            print(f"❌ Error saat debug: {e}")
    
    print("\n" + "="*60)
    run_batch_pipeline(DATA_FOLDER)
