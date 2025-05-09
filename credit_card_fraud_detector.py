# credit_card_fraud_detector.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_curve, 
    roc_auc_score,
    precision_recall_curve, 
    average_precision_score,
    recall_score,
    f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import time
import warnings
import os
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
warnings.filterwarnings('ignore')

# ========================================
# 1. Configuration and Setup
# ========================================
RANDOM_STATE = 42
MAX_SAMPLES = 10000
ENABLE_SHAP = True
ENABLE_HYBRID = True

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Set random seed
np.random.seed(RANDOM_STATE)

# ========================================
# 2. Helper Functions
# ========================================
def save_plot(fig, filename):
    """Save plot to output folder with enhanced error handling"""
    path = os.path.join(output_dir, filename)
    try:
        # Make sure the figure is complete before saving
        fig.tight_layout(pad=2.0)
        plt.savefig(path, dpi=150, bbox_inches='tight', format='png')
        plt.close(fig)
        
        # Verify the file was created and is a valid image
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"✅ Saved plot to {filename}")
            return path
        else:
            print(f"⚠️ Failed to save {filename}: File is empty or missing")
            return None
    except Exception as e:
        print(f"⚠️ Failed to save {filename}: {str(e)}")
        # Try to close the figure anyway to prevent resource leaks
        try:
            plt.close(fig)
        except:
            pass
        return None

def debug_plot_files(plot_paths):
    """Check if plot files exist and are valid images"""
    print("\n=== DEBUGGING IMAGE FILES ===")
    for i, path in enumerate(plot_paths):
        if path is None:
            print(f"Path {i}: None")
            continue
            
        print(f"Path {i}: {path}")
        if not os.path.exists(path):
            print(f"  ❌ File does not exist")
            continue
            
        try:
            # Try to verify image is valid
            with Image.open(path) as img:
                print(f"  ✅ Valid image: {img.format}, {img.size}")
        except Exception as e:
            print(f"  ❌ Not a valid image: {str(e)}")
    print("===========================\n")
    return [p for p in plot_paths if p and os.path.exists(p)]

# Now, let's create a simpler viewer function
def show_results_simple(plot_paths):
    """A simple image viewer that avoids complex Tkinter interactions"""
    valid_paths = debug_plot_files(plot_paths)
    
    if not valid_paths:
        print("⚠️ No valid images to display")
        return
        
    print(f"📊 Found {len(valid_paths)} valid images to display")
    
    try:
        # Simple approach: show images one by one without maintaining a GUI
        for path in valid_paths:
            print(f"\nDisplaying: {os.path.basename(path)}")
            print("(Close the image window to continue to the next one)")
            img = Image.open(path)
            img.show()  # This uses the system's default image viewer
    except Exception as e:
        print(f"⚠️ Error displaying images: {str(e)}")
        
    print("✅ Image display complete")

# Alternative approach using matplotlib instead of tkinter
def show_results_professional(plot_paths):
    """A professional integrated image viewer using Matplotlib in Tkinter"""
    valid_paths = debug_plot_files(plot_paths)
    
    if not valid_paths:
        print("⚠️ No valid images to display")
        return
        
    print(f"📊 Displaying {len(valid_paths)} visualizations in dashboard viewer")
    
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    
    # Initialize Tkinter window
    root = tk.Tk()
    root.title("Credit Card Fraud Detection Dashboard")
    root.geometry("1200x800")
    root.configure(bg="#f0f0f0")  # Light gray background
    
    # Flag to track whether we need to exit the application
    dashboard_closed = False
    
    # Add title and styling
    title_frame = ttk.Frame(root)
    title_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
    
    title_label = ttk.Label(
        title_frame, 
        text="Credit Card Fraud Detection Results", 
        font=("Arial", 16, "bold")
    )
    title_label.pack(side=tk.LEFT)
    
    # Style configuration
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 10))
    style.configure("TLabel", font=("Arial", 11))
    
    # Create a frame for the current image index display
    index_frame = ttk.Frame(title_frame)
    index_frame.pack(side=tk.RIGHT, padx=10)
    
    index_var = tk.StringVar()
    index_label = ttk.Label(index_frame, textvariable=index_var)
    index_label.pack()
    
    # Frame for matplotlib figure
    fig_frame = ttk.Frame(root)
    fig_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
    
    # Create matplotlib figure with larger size
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.tight_layout(pad=3)
    
    # Embed matplotlib figure in tkinter
    canvas = FigureCanvasTkAgg(fig, master=fig_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Add matplotlib navigation toolbar (zoom, pan, save, etc.)
    toolbar_frame = ttk.Frame(root)
    toolbar_frame.pack(fill=tk.X, padx=20)
    toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
    toolbar.update()
    
    # Variable to track current image
    current_img = [0]
    
    def show_image(index):
        """Display image at the given index"""
        ax.clear()
        path = valid_paths[index]
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis('off')
        filename = os.path.basename(path)
        
        # Update window title and index display
        root.title(f"Fraud Detection - {filename}")
        index_var.set(f"Visualization {index+1} of {len(valid_paths)}")
        
        # Update current index 
        current_img[0] = index
        
        # Refresh canvas
        canvas.draw()
    
    def exit_dashboard():
        """Properly close the dashboard and set the exit flag"""
        nonlocal dashboard_closed
        dashboard_closed = True
        root.quit()  # This stops the mainloop
        root.destroy()  # This removes the window
    
    # Navigation buttons with better styling
    nav_frame = ttk.Frame(root)
    nav_frame.pack(fill=tk.X, padx=20, pady=(10, 20))
    
    # Left side - navigation buttons
    btn_frame = ttk.Frame(nav_frame)
    btn_frame.pack(side=tk.LEFT)
    
    ttk.Button(
        btn_frame, 
        text="◀ Previous",
        command=lambda: show_image((current_img[0]-1) % len(valid_paths)),
        width=12
    ).grid(row=0, column=0, padx=5)
    
    ttk.Button(
        btn_frame, 
        text="Next ▶",
        command=lambda: show_image((current_img[0]+1) % len(valid_paths)),
        width=12
    ).grid(row=0, column=1, padx=5)
    
    # Right side - exit button
    exit_frame = ttk.Frame(nav_frame)
    exit_frame.pack(side=tk.RIGHT)
    
    ttk.Button(
        exit_frame, 
        text="Close Dashboard",
        command=exit_dashboard,  # Use our new exit function
        width=15
    ).pack(padx=5)
    
    # Keyboard navigation
    root.bind('<Left>', lambda _: show_image((current_img[0]-1) % len(valid_paths)))
    root.bind('<Right>', lambda _: show_image((current_img[0]+1) % len(valid_paths)))
    root.bind('<Escape>', lambda _: exit_dashboard())
    
    # Handle window close button (X)
    root.protocol("WM_DELETE_WINDOW", exit_dashboard)
    
    # Display first image
    if valid_paths:
        show_image(0)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Start main loop with proper error handling
    try:
        root.mainloop()
    except Exception as e:
        print(f"⚠️ Dashboard error: {str(e)}")
        try:
            root.destroy()
        except:
            pass
        dashboard_closed = True  # Make sure we mark it as closed
    
    # Ensure all matplotlib resources are properly cleaned up
    try:
        plt.close(fig)
    except:
        pass
        
    print("✅ Dashboard viewer closed")
    return dashboard_closed  # Return whether the dashboard was closed properly

# ========================================
# 3. Main Pipeline
# ========================================
def main():
    print("🚀 Starting Credit Card Fraud Detection System")
    start_time = time.time()
    plot_paths = []
    
    # 1. Data Loading
    print("\n📊 Loading and preprocessing data...")
    def load_data(path='creditcard.csv'):
        dtypes = {f'V{i}': 'float32' for i in range(1, 29)}
        dtypes.update({'Time': 'float32', 'Amount': 'float32', 'Class': 'int8'})
        return pd.read_csv(path, dtype=dtypes)

    try:
        df = load_data()
        print(f"✅ Dataset loaded: {df.shape[0]:,} transactions")
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        exit()

    # 2. Feature Engineering
    df['Hour'] = (df['Time'] % (24 * 3600)) // 3600
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour']/24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour']/24)
    df = df.drop(['Time', 'Hour'], axis=1)
    
    # 3. Data Scaling
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    
    # 4. Train-Test Split
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # 5. Handle Class Imbalance
    print("\n🔧 Balancing class distribution...")
    sampler = Pipeline([
        ('smote', SMOTE(sampling_strategy=0.1, random_state=RANDOM_STATE)),
        ('under', RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE))
    ])
    X_res, y_res = sampler.fit_resample(X_train, y_train)

    # 6. Model Training
    print("\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=50,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_res, y_res)
    print("✅ Model trained!")

    # 7. Threshold Optimization
    def calculate_fraud_cost(y_true, y_pred):
        FP_cost = 10
        FN_cost = 500
        cm = confusion_matrix(y_true, y_pred)
        return cm[0][1] * FP_cost + cm[1][0] * FN_cost

    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0, 1, 50)
    costs = [calculate_fraud_cost(y_test, (y_prob >= t).astype(int)) for t in thresholds]
    optimal_threshold = thresholds[np.argmin(costs)]
    y_pred = (y_prob >= optimal_threshold).astype(int)

    print(f"\n🎯 Optimal Threshold: {optimal_threshold:.3f}")
    print(f"💵 Minimum Business Cost: ${calculate_fraud_cost(y_test, y_pred):,}")

    # 8. SHAP Explanations
    if ENABLE_SHAP:
        print("\n🔍 Generating model explanations...")
        try:
            import shap
            background = shap.sample(X_train, min(100, len(X_train)))
            fraud_sample = X_test[y_test == 1].sample(1, random_state=RANDOM_STATE)
            proba = model.predict_proba(fraud_sample)[0][1]
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(fraud_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            plt.figure(figsize=(10, 5))
            shap.summary_plot(shap_values, fraud_sample, plot_type='bar', show=False, max_display=10)
            plt.title(f'Fraud Explanation (Probability: {proba:.1%})')
            plot_paths.append(save_plot(plt.gcf(), 'fraud_explanation.png'))
            print(f"✅ SHAP analysis saved! (Fraud probability: {proba:.1%})")
        except Exception as e:
            print(f"⚠️ SHAP failed: {str(e)}")

    # 9. Anomaly Detection
    if ENABLE_HYBRID:
        print("\n🕵️ Adding anomaly detection...")
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(X_train.sample(MAX_SAMPLES, random_state=RANDOM_STATE))
        X_test['Anomaly_Score'] = lof.decision_function(X_test)
        print(f"🔎 Anomaly scores added (mean: {X_test['Anomaly_Score'].mean():.2f})")

    # 10. Visualizations
    print("\n📊 Creating dashboard visuals...")
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('ggplot')

    # ROC Curve
    plt.figure(figsize=(10, 5))
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, y_prob):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plot_paths.append(save_plot(plt.gcf(), 'roc_curve.png'))

    # Feature Importance
    plt.figure(figsize=(10, 6))
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(15).plot(kind='barh')
    plt.title('Top 15 Important Features')
    plot_paths.append(save_plot(plt.gcf(), 'feature_importance.png'))
    print("✅ Dashboard visuals saved!")

    # 11. Simulation
    print("\n🎮 Running fraud simulation...")
    def safe_predict(model, sample):
        expected_features = model.feature_names_in_
        return model.predict_proba(sample.reindex(columns=expected_features, fill_value=0))[0][1]

    from tqdm import tqdm
    print("\n=== Transaction Monitor ===")
    for i in tqdm(range(5), desc="Processing"):
        sample = X_test.drop(columns=['Anomaly_Score'], errors='ignore').sample(1)
        proba = safe_predict(model, sample)
        status = "🚨 FRAUD" if proba > optimal_threshold else "✅ Genuine"
        print(f"Transaction {i+1}: {status} (p={proba:.1%})")
        time.sleep(0.5)

    # 12. Final Report
    print("\n📝 Generating final report...")
    execution_time = (time.time() - start_time) / 60
    report = f"""
# CREDIT CARD FRAUD DETECTION REPORT

## Performance Summary
- Optimal Threshold: {optimal_threshold:.3f}
- Business Cost: ${calculate_fraud_cost(y_test, y_pred):,}
- Fraud Detection Rate: {recall_score(y_test, y_pred):.1%}
- False Positive Rate: {fpr[1]:.1%}

## Key Metrics
{classification_report(y_test, y_pred, target_names=['Genuine', 'Fraud'])}

## System Info
- Execution Time: {execution_time:.1f} minutes
- Samples Processed: {len(df):,}
- Features Used: {X.shape[1]}

Generated on {time.strftime('%Y-%m-%d %H:%M')}
"""
    with open(os.path.join(output_dir, 'fraud_report.md'), 'w') as f:
        f.write(report)

    # Show all results
    print(f"\n🎉 Project completed in {execution_time:.1f} minutes!")
    print(f"📁 All outputs saved to: {os.path.abspath(output_dir)}")
    
    # Use our debugging function first to check files
    valid_plots = debug_plot_files(plot_paths)
    
    # Use professional dashboard viewer and ensure we exit properly afterward
    if valid_plots:
        try:
            dashboard_closed = show_results_professional(valid_plots)
            if not dashboard_closed:
                print("⚠️ Dashboard may not have closed properly")
        except Exception as e:
            print(f"⚠️ Error displaying dashboard: {str(e)}")
    else:
        print("⚠️ No valid plot files were created. Check the output directory manually.")
    
    print("✅ Program execution completed!")

if __name__ == "__main__":
    main()