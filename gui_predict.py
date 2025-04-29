import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar
from catboost import CatBoostClassifier
import pandas as pd
from datetime import date, datetime
import time

# Mappings and options
bmi_mapping = {'Low': 0, 'Normal': 1, 'High': 2}
cholesterol_mapping = {'Normal': 0, 'High': 1}
treatment_mapping = {'Surgery': 0, 'Chemotherapy': 1, 'Radiation': 2, 'Combined': 3}
gender_options = ['Male', 'Female']
country_options = ['USA', 'UK', 'India', 'Other']
cancer_stage_options = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
yes_no_options = ['Yes', 'No']
smoking_status_options = ['Never', 'Current', 'Former']
treatment_type_options = ['Surgery', 'Chemotherapy', 'Radiation', 'Combined']

class ModernGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lung Cancer Survival Prediction System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Dark mode state
        self.dark_mode = False
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        self.style.configure('Header.TLabel', background='#f0f0f0', font=('Helvetica', 16, 'bold'))
        self.style.configure('Section.TLabel', background='#f0f0f0', font=('Helvetica', 12, 'bold'))
        self.style.configure('TButton', font=('Helvetica', 12, 'bold'))
        self.style.configure('Predict.TButton', 
                           background='#4CAF50', 
                           foreground='white',
                           padding=10)
        
        # Create main container
        self.main_container = ttk.Frame(root, padding="20")
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create header
        self.create_header()
        
        # Create toolbar
        self.create_toolbar()
        
        # Create scrollable frame
        self.create_scrollable_frame()
        
        # Create form sections
        self.create_form_sections()
        
        # Create status bar
        self.create_status_bar()
        
        # Load model
        self.load_model()
        
        # Create tooltips
        self.create_tooltips()
        
    def create_header(self):
        header_frame = ttk.Frame(self.main_container)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        header_label = ttk.Label(header_frame, 
                               text="Lung Cancer Survival Prediction System",
                               style='Header.TLabel')
        header_label.pack()
        
        subtitle_label = ttk.Label(header_frame,
                                 text="Enter patient information below to predict survival probability",
                                 style='TLabel')
        subtitle_label.pack(pady=(5, 0))
        
    def create_toolbar(self):
        toolbar_frame = ttk.Frame(self.main_container)
        toolbar_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Dark mode toggle
        self.dark_mode_var = tk.BooleanVar(value=False)
        dark_mode_check = ttk.Checkbutton(
            toolbar_frame, 
            text="Dark Mode", 
            variable=self.dark_mode_var,
            command=self.toggle_dark_mode
        )
        dark_mode_check.pack(side=tk.LEFT, padx=5)
        
        # Reset button
        reset_button = ttk.Button(
            toolbar_frame,
            text="Reset Form",
            command=self.reset_form
        )
        reset_button.pack(side=tk.RIGHT, padx=5)
        
    def create_scrollable_frame(self):
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(self.main_container, bg='#f0f0f0', highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=1160)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
    def create_form_sections(self):
        # Create two-column layout
        columns_frame = ttk.Frame(self.scrollable_frame)
        columns_frame.pack(fill=tk.X, pady=10)
        
        # Left column
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Right column
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Personal Information Section (Left)
        self.create_section(left_column, "Personal Information", [
            ("Patient ID:", "id_entry", "entry", "Enter unique patient identifier"),
            ("Age:", "age_entry", "entry", "Enter patient's age (0-120)"),
            ("Gender:", "gender_var", "dropdown", gender_options, "Select patient's gender"),
            ("Country:", "country_var", "dropdown", country_options, "Select patient's country")
        ])
        
        # Medical Information Section (Left)
        self.create_section(left_column, "Medical Information", [
            ("Cancer Stage:", "stage_var", "dropdown", cancer_stage_options, "Select cancer stage"),
            ("Family History:", "family_history_var", "dropdown", yes_no_options, "Does patient have family history of cancer?"),
            ("Smoking Status:", "smoking_status_var", "dropdown", smoking_status_options, "Select patient's smoking status"),
            ("BMI:", "bmi_var", "dropdown", list(bmi_mapping.keys()), "Select patient's BMI category"),
            ("Cholesterol Level:", "cholesterol_var", "dropdown", list(cholesterol_mapping.keys()), "Select patient's cholesterol level")
        ])
        
        # Health Conditions Section (Right)
        self.create_section(right_column, "Health Conditions", [
            ("Hypertension:", "hypertension_var", "dropdown", yes_no_options, "Does patient have hypertension?"),
            ("Asthma:", "asthma_var", "dropdown", yes_no_options, "Does patient have asthma?"),
            ("Cirrhosis:", "cirrhosis_var", "dropdown", yes_no_options, "Does patient have cirrhosis?"),
            ("Other Cancer:", "other_cancer_var", "dropdown", yes_no_options, "Does patient have other types of cancer?")
        ])
        
        # Treatment Information Section (Right)
        self.create_section(right_column, "Treatment Information", [
            ("Treatment Type:", "treatment_type_var", "dropdown", treatment_type_options, "Select treatment type"),
            ("Diagnosis Date:", "diagnosis_date", "calendar", None, "Select date of diagnosis"),
            ("End Treatment Date:", "end_treatment_date", "calendar", None, "Select date treatment ended")
        ])
        
        # Predict Button and Result (Bottom)
        self.create_prediction_section()
        
    def create_section(self, parent, title, fields):
        # Create section frame
        section_frame = ttk.Frame(parent)
        section_frame.pack(fill=tk.X, pady=10)
        
        # Section title
        title_label = ttk.Label(section_frame, text=title, style='Section.TLabel')
        title_label.pack(anchor='w', pady=(0, 10))
        
        # Create fields
        for field_info in fields:
            label_text, var_name, field_type, *args = field_info
            tooltip_text = args[-1] if len(args) > 1 else None
            field_values = args[0] if args else None
            
            field_frame = ttk.Frame(section_frame)
            field_frame.pack(fill=tk.X, pady=5)
            
            label = ttk.Label(field_frame, text=label_text, width=15)
            label.pack(side="left")
            
            if field_type == "entry":
                entry = ttk.Entry(field_frame)
                entry.pack(side="left", expand=True, fill="x", padx=(10, 0))
                setattr(self, var_name, entry)
                
                # Add validation
                if label_text == "Age:":
                    entry.bind('<FocusOut>', lambda e, entry=entry: self.validate_age(entry))
                
            elif field_type == "dropdown":
                var = tk.StringVar()
                combo = ttk.Combobox(field_frame, textvariable=var, values=field_values, state='readonly')
                combo.pack(side="left", expand=True, fill="x", padx=(10, 0))
                setattr(self, var_name, var)
                
            elif field_type == "calendar":
                calendar_frame = ttk.Frame(field_frame)
                calendar_frame.pack(side="left", expand=True, fill="x", padx=(10, 0))
                
                calendar = Calendar(calendar_frame, selectmode='day', date_pattern='y-mm-dd',
                                  background='#4CAF50', foreground='white',
                                  selectbackground='#2E7D32', selectforeground='white',
                                  normalbackground='#f0f0f0', normalforeground='black',
                                  weekendbackground='#E8F5E9', weekendforeground='black',
                                  othermonthbackground='#E0E0E0', othermonthforeground='gray')
                calendar.pack(side="left", expand=True, fill="x")
                setattr(self, var_name, calendar)
                
            # Store tooltip info
            if tooltip_text:
                setattr(self, f"{var_name}_tooltip", tooltip_text)
                
    def create_prediction_section(self):
        # Create prediction section
        prediction_frame = ttk.Frame(self.scrollable_frame)
        prediction_frame.pack(fill=tk.X, pady=20)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            prediction_frame,
            variable=self.progress_var,
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        # Predict button
        predict_button = ttk.Button(
            prediction_frame,
            text="Predict Survival",
            command=self.predict,
            style='Predict.TButton'
        )
        predict_button.pack(pady=10)
        
        # Result label
        self.result_label = ttk.Label(
            prediction_frame,
            text="",
            font=('Helvetica', 14, 'bold'),
            style='Header.TLabel'
        )
        self.result_label.pack(pady=10)
        
    def create_status_bar(self):
        self.status_bar = ttk.Label(
            self.main_container,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def create_tooltips(self):
        # Create tooltips for all fields
        for attr_name in dir(self):
            if attr_name.endswith('_tooltip'):
                widget_name = attr_name.replace('_tooltip', '')
                tooltip_text = getattr(self, attr_name)
                
                # Get the actual widget instead of the StringVar
                widget = getattr(self, widget_name, None)
                if isinstance(widget, tk.StringVar):
                    # Find the associated combobox widget
                    for child in self.scrollable_frame.winfo_children():
                        if isinstance(child, ttk.Frame):
                            for grandchild in child.winfo_children():
                                if isinstance(grandchild, ttk.Frame):
                                    for great_grandchild in grandchild.winfo_children():
                                        if isinstance(great_grandchild, ttk.Combobox) and great_grandchild.cget('textvariable') == str(widget):
                                            self.create_tooltip(great_grandchild, tooltip_text)
                                            break
                elif widget and hasattr(widget, 'bind'):
                    self.create_tooltip(widget, tooltip_text)
                    
    def create_tooltip(self, widget, text):
        def show_tooltip(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(tooltip, text=text, background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
            
            def hide_tooltip():
                tooltip.destroy()
                
            widget.tooltip = tooltip
            widget.bind('<Leave>', lambda e: hide_tooltip())
            tooltip.bind('<Leave>', lambda e: hide_tooltip())
            
        widget.bind('<Enter>', show_tooltip)
        
    def validate_age(self, entry):
        try:
            age = int(entry.get())
            if not (0 <= age <= 120):
                messagebox.showwarning("Invalid Age", "Age must be between 0 and 120")
                entry.config(foreground='red')
            else:
                entry.config(foreground='black')
        except ValueError:
            messagebox.showwarning("Invalid Age", "Age must be a number")
            entry.config(foreground='red')
            
    def toggle_dark_mode(self):
        self.dark_mode = self.dark_mode_var.get()
        
        if self.dark_mode:
            # Dark mode colors
            bg_color = '#2d2d2d'
            fg_color = '#ffffff'
            self.root.configure(bg=bg_color)
            self.style.configure('TFrame', background=bg_color)
            self.style.configure('TLabel', background=bg_color, foreground=fg_color)
            self.style.configure('Header.TLabel', background=bg_color, foreground=fg_color)
            self.style.configure('Section.TLabel', background=bg_color, foreground=fg_color)
            self.canvas.configure(bg=bg_color)
        else:
            # Light mode colors
            bg_color = '#f0f0f0'
            fg_color = '#000000'
            self.root.configure(bg=bg_color)
            self.style.configure('TFrame', background=bg_color)
            self.style.configure('TLabel', background=bg_color, foreground=fg_color)
            self.style.configure('Header.TLabel', background=bg_color, foreground=fg_color)
            self.style.configure('Section.TLabel', background=bg_color, foreground=fg_color)
            self.canvas.configure(bg=bg_color)
            
    def reset_form(self):
        # Reset all entry fields
        self.id_entry.delete(0, tk.END)
        self.age_entry.delete(0, tk.END)
        self.age_entry.config(foreground='black')
        
        # Reset all dropdowns
        for attr_name in dir(self):
            if attr_name.endswith('_var'):
                var = getattr(self, attr_name)
                if isinstance(var, tk.StringVar):
                    var.set('')
                    
        # Reset calendars to today
        today = date.today().strftime('%Y-%m-%d')
        self.diagnosis_date.selection_set(today)
        self.end_treatment_date.selection_set(today)
        
        # Reset result
        self.result_label.config(text="")
        
        # Update status
        self.status_bar.config(text="Form reset")
        
    def load_model(self):
        try:
            self.status_bar.config(text="Loading model...")
            self.model = CatBoostClassifier()
            self.model.load_model('lung_cancer_survival_model.cbm')
            self.status_bar.config(text="Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model = None
            self.status_bar.config(text="Failed to load model")
            
    def predict(self):
        try:
            self.status_bar.config(text="Validating inputs...")
            
            # Get values
            patient_id = self.id_entry.get()
            
            # Validate age
            try:
                age = int(self.age_entry.get())
                if not (0 <= age <= 120):
                    messagebox.showerror("Error", "Age must be between 0 and 120")
                    return
            except ValueError:
                messagebox.showerror("Error", "Age must be a number")
                return
                
            gender = self.gender_var.get()
            country = self.country_var.get()
            stage = self.stage_var.get()
            family_history = self.family_history_var.get()
            smoking_status = self.smoking_status_var.get()
            bmi = self.bmi_var.get()
            cholesterol = self.cholesterol_var.get()
            hypertension = self.hypertension_var.get()
            asthma = self.asthma_var.get()
            cirrhosis = self.cirrhosis_var.get()
            other_cancer = self.other_cancer_var.get()
            treatment_type = self.treatment_type_var.get()
            
            # Validate all fields are filled
            if not all([patient_id, age, gender, country, stage, family_history, 
                       smoking_status, bmi, cholesterol, hypertension, asthma, 
                       cirrhosis, other_cancer, treatment_type]):
                messagebox.showerror("Error", "Please fill in all fields")
                return
                
            # Get dates and convert to datetime objects
            try:
                diagnosis_date_str = self.diagnosis_date.selection_get().strftime('%Y-%m-%d')
                diagnosis_date = datetime.strptime(diagnosis_date_str, '%Y-%m-%d').date()
                
                end_treatment_date_str = self.end_treatment_date.selection_get().strftime('%Y-%m-%d')
                end_treatment_date = datetime.strptime(end_treatment_date_str, '%Y-%m-%d').date()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid date format: {str(e)}")
                return
                
            # Validate dates
            if end_treatment_date < diagnosis_date:
                messagebox.showerror("Error", "End treatment date cannot be before diagnosis date")
                return
                
            # Calculate days
            days_since_diagnosis = (date.today() - diagnosis_date).days
            days_since_end_treatment = (date.today() - end_treatment_date).days
            
            # Update progress
            self.progress_var.set(20)
            self.status_bar.config(text="Preparing data...")
            self.root.update()
            
            # Prepare data
            data = {
                'age': [age],
                'gender': [gender],
                'country': [country],
                'cancer_stage': [stage],
                'family_history': [1 if family_history == 'Yes' else 0],
                'smoking_status': [smoking_status],
                'bmi': [bmi_mapping[bmi]],
                'cholesterol_level': [cholesterol_mapping[cholesterol]],
                'hypertension': [1 if hypertension == 'Yes' else 0],
                'asthma': [1 if asthma == 'Yes' else 0],
                'cirrhosis': [1 if cirrhosis == 'Yes' else 0],
                'other_cancer': [1 if other_cancer == 'Yes' else 0],
                'treatment_type': [treatment_mapping[treatment_type]],
                'days_since_diagnosis': [days_since_diagnosis],
                'days_since_end_treatment': [days_since_end_treatment]
            }
            
            # Update progress
            self.progress_var.set(40)
            self.status_bar.config(text="Converting data...")
            self.root.update()
            
            # Convert to DataFrame
            input_df = pd.DataFrame(data)
            
            # Update progress
            self.progress_var.set(60)
            self.status_bar.config(text="Making prediction...")
            self.root.update()
            
            # Make prediction
            if self.model is None:
                messagebox.showerror("Error", "Model not loaded")
                return
                
            # Simulate processing time for better UX
            time.sleep(0.5)
            
            prediction = self.model.predict(input_df)[0]
            
            # Update progress
            self.progress_var.set(100)
            self.status_bar.config(text="Prediction complete")
            self.root.update()
            
            # Update result with animation
            self.result_label.config(
                text="✅ Patient SURVIVED ✅" if prediction == 1 else "❌ Patient DID NOT SURVIVE ❌",
                foreground="green" if prediction == 1 else "red"
            )
            
            # Show detailed message
            if prediction == 1:
                messagebox.showinfo("Prediction Result", 
                    "The model predicts that the patient is likely to survive.\n\n"
                    "This prediction is based on the provided medical information and treatment history.")
            else:
                messagebox.showinfo("Prediction Result", 
                    "The model predicts that the patient may not survive.\n\n"
                    "This prediction is based on the provided medical information and treatment history.\n"
                    "Please consult with healthcare professionals for proper medical advice.")
            
            # Reset progress bar after a delay
            self.root.after(2000, lambda: self.progress_var.set(0))
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_bar.config(text="Error during prediction")
            self.progress_var.set(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernGUI(root)
    root.mainloop() 