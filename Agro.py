import tkinter as tk
from tkinter import ttk, scrolledtext
import joblib
import pandas as pd

# Load updated models and label encoders
crop_model = joblib.load('ensemble_crop_model.pkl')
crop_label_encoders = joblib.load('crop_label_encoders.pkl')
fertilizer_model = joblib.load('fertilizer_model.pkl')
fertilizer_label_encoders = joblib.load('fertilizer_label_encoders.pkl')


class AgroAidGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Agro Aid - Crop & Fertilizer Recommendation")
        self.root.geometry("800x600")
        self.root.configure(bg='#1a1a1a')

        self.bot = AgroAidBot()
        self.bot.display_message = self.display_bot_message

        self.setup_gui()

    def setup_gui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#1a1a1a')
        style.configure('Dark.TButton', background='#444444', foreground='white', padding=8)
        style.map('Dark.TButton', background=[('active', '#666666')])
        style.configure('Input.TEntry', fieldbackground='#333333', foreground='white', insertcolor='white')

        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.chat_display = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=70, height=20,
            font=('Arial', 11), bg='#1a1a1a', fg='white', insertbackground='white'
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.chat_display.tag_configure('bot', foreground='#00ff00', font=('Arial', 11, 'bold'))
        self.chat_display.tag_configure('user', foreground='#ffffff')

        input_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.input_field = ttk.Entry(input_frame, font=('Arial', 11), style='Input.TEntry')
        self.input_field.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.input_field.bind('<Return>', lambda e: self.send_message())

        send_button = ttk.Button(input_frame, text="Send", style='Dark.TButton', command=self.send_message)
        send_button.pack(side=tk.RIGHT)

        options_frame = ttk.Frame(main_frame, style='Dark.TFrame')
        options_frame.pack(fill=tk.X)

        options = [("Crop Prediction", "1"), ("Fertilizer Recommendation", "2")]
        for text, value in options:
            btn = ttk.Button(options_frame, text=text, style='Dark.TButton',
                             command=lambda v=value: self.quick_option(v))
            btn.pack(side=tk.LEFT, padx=5)

        self.display_bot_message("Hello! I am Agro Aid. How can I help you today?\n")
        self.display_bot_message("Please choose an option:\n1) Crop prediction\n2) Fertilizer recommendation\nType 'quit' to exit")

    def display_bot_message(self, message):
        self.chat_display.insert(tk.END, f"🤖 Bot: {message}\n", 'bot')
        self.chat_display.see(tk.END)

    def display_user_message(self, message):
        self.chat_display.insert(tk.END, f"👤 You: {message}\n", 'user')
        self.chat_display.see(tk.END)

    def send_message(self):
        message = self.input_field.get().strip()
        if message:
            self.input_field.delete(0, tk.END)
            self.display_user_message(message)
            if message.lower() == 'quit':
                self.display_bot_message("Goodbye! 👋")
                self.root.after(1000, self.root.destroy)
                return
            self.bot.process_input(message)

    def quick_option(self, option):
        self.input_field.delete(0, tk.END)
        self.input_field.insert(0, option)
        self.send_message()

    def run(self):
        self.root.mainloop()


class AgroAidBot:
    def __init__(self):
        self.state = 'main_menu'
        self.inputs = {}
        self.step_index = 0
        self.display_message = print

        self.crop_steps = [
            'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity',
            'pH_Value', 'Rainfall'
        ]
        self.ferti_steps = [
            'Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop',
            'Nitrogen', 'Potassium', 'Phosphorus'
        ]

    def process_input(self, message):
        if self.state == 'main_menu':
            if message == '1':
                self.state = 'crop_prediction'
                self.step_index = 0
                self.inputs.clear()
                self.display_message("You've selected Crop Prediction.")
                self.display_message(f"Please enter {self.crop_steps[self.step_index]}:")
            elif message == '2':
                self.state = 'fertilizer_start'
                self.step_index = 0
                self.inputs.clear()
                self.display_message("You've selected Fertilizer Recommendation.")
                self.display_message(f"Please enter {self.ferti_steps[self.step_index]}:")
            else:
                self.display_message("❌ Invalid option. Please choose between 1-2.")

        elif self.state == 'crop_prediction':
            key = self.crop_steps[self.step_index]
            self.inputs[key] = message
            self.step_index += 1
            if self.step_index < len(self.crop_steps):
                self.display_message(f"Please enter {self.crop_steps[self.step_index]}:")
            else:
                self.run_crop_prediction()
                self.state = 'main_menu'
                self.display_message("\nBack to main menu. Please choose an option:")

        elif self.state == 'fertilizer_start':
            key = self.ferti_steps[self.step_index]
            self.inputs[key] = message
            self.step_index += 1
            if self.step_index < len(self.ferti_steps):
                self.display_message(f"Please enter {self.ferti_steps[self.step_index]}:")
            else:
                self.run_fertilizer_prediction()
                self.state = 'main_menu'
                self.display_message("\nBack to main menu. Please choose an option:")

    def run_crop_prediction(self):
        try:
            for key in self.crop_steps:
                self.inputs[key] = float(self.inputs[key])

            df = pd.DataFrame([self.inputs])
            pred = crop_model.predict(df)[0]

            predicted_soil = crop_label_encoders['Soil_Type'].inverse_transform([pred[0]])[0]
            predicted_variety = crop_label_encoders['Variety'].inverse_transform([pred[1]])[0]

            self.display_message(f"✅ Suitable Soil Type: {predicted_soil}")
            self.display_message(f"🌾 Recommended Crop Variety: {predicted_variety}")
        except Exception as e:
            self.display_message(f"❌ Error during crop prediction: {str(e)}")

    def run_fertilizer_prediction(self):
        try:
            for key in ['Temperature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorus']:
                self.inputs[key] = float(self.inputs[key])

            soil_encoded = fertilizer_label_encoders['Soil_Type'].transform([self.inputs['Soil_Type']])[0]
            crop_encoded = fertilizer_label_encoders['Crop'].transform([self.inputs['Crop']])[0]

            fert_input = [
                int(self.inputs['Temperature']),
                int(self.inputs['Humidity']),
                int(self.inputs['Moisture']),
                soil_encoded,
                crop_encoded,
                int(self.inputs['Nitrogen']),
                int(self.inputs['Potassium']),
                int(self.inputs['Phosphorus'])
            ]

            fert_result = fertilizer_model.predict([fert_input])[0]
            fert_name = fertilizer_label_encoders['FertilizerName'].inverse_transform([fert_result])[0]
            self.display_message(f"💡 Recommended Fertilizer: {fert_name}")
        except Exception as e:
            self.display_message(f"❌ Error during fertilizer recommendation: {str(e)}")


if __name__ == '__main__':
    AgroAidGUI().run()
