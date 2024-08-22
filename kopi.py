
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class CoffeeQualityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coffee Quality Analyzer")
        
        self.label = tk.Label(root, text="Load your training data:")
        self.label.pack(pady=10)
        
        self.load_button = tk.Button(root, text="Load Data", command=self.load_data)
        self.load_button.pack(pady=5)
        
        self.train_button = tk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=5)
        
        self.predict_button = tk.Button(root, text="Predict Quality", command=self.predict_quality)
        self.predict_button.pack(pady=5)
        
        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

        self.data = None
        self.model = None

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            messagebox.showinfo("Info", "Data loaded successfully!")
        else:
            messagebox.showwarning("Warning", "No file selected.")
    
    def train_model(self):
        if self.data is not None:
            try:
                X = self.data.drop("Quality", axis=1)
                y = self.data["Quality"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                self.model = DecisionTreeClassifier()
                self.model.fit(X_train, y_train)
                
                y_pred = self.model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                messagebox.showinfo("Model Training", f"Model trained successfully! Accuracy: {accuracy:.2f}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {e}")
        else:
            messagebox.showwarning("Warning", "Please load data first.")
    
    def predict_quality(self):
        if self.model is not None:

            sample = self.data.drop("Quality", axis=1).iloc[0].values.reshape(1, -1)
            prediction = self.model.predict(sample)
            self.result_label.config(text=f"Predicted Quality: {prediction[0]}")
        else:
            messagebox.showwarning("Warning", "Please train the model first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = CoffeeQualityApp(root)
    root.mainloop()
