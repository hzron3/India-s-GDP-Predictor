from datetime import datetime
import os
from tkinter import filedialog, Tk
import pandas as pd
from reportlab.lib.utils import ImageReader
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from tkinter import *
from tkinter import filedialog, messagebox
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Create a function to upload the dataset
def browse_files(message_label=None):
    while True:
        filename = filedialog.askopenfilename(initialdir="/", title="Select a File",
                                              filetypes=(("CSV Files", "*.csv*"),))
        if filename.endswith('.csv'):
            file_path.set(filename)
            print(file_path.get())  # add this line to print the file path
            if message_label is not None:
                message_label.config(text="Data cleaned and split successfully!", fg="green")
            break
        else:
            # Show a popup error message
            messagebox.showerror("Error", "Please select a CSV file.")
# Define a function to clean and split data
data_cleaned_and_split = False


def clean_split_data(message_label=None):
    global data_cleaned_and_split
    if not file_path.get():
        messagebox.showerror("Error", "Please select a CSV file.")
        if message_label is not None:
            message_label.config(text="Please select a CSV file first!", fg="red")
        return None, None, None, None

    try:
        # Read the dataset
        data = pd.read_csv(file_path.get())
        print("Data shape:", data.shape)

        # Clean the data
        print("Number of missing values before cleaning:", data.isna().sum().sum())
        data.dropna(inplace=True)  # Drop rows with missing values
        data = data.apply(lambda x: x.str.replace(',', '.') if x.dtype == 'object' else x)  # Replace commas with periods
        data = data.apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()  # Convert string to float and remove invalid values
        print("Number of missing values after cleaning:", data.isna().sum().sum())

        # Split the data into training and testing sets
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)

        # Check if a message label was provided
        if message_label is not None:
            messagebox.showinfo("Data Cleaning", "Data cleaned and split successfully!")
            message_label.config(text="Data cleaned and split successfully!", fg="green")

        # Set the variable to True indicating that the data has been cleaned and split
        data_cleaned_and_split = True

        return X_train, X_test, y_train, y_test
    except Exception as e:
        # Check if a message label was provided
        if message_label is not None:
            messagebox.showerror("Data Cleaning Error", str(e))
            message_label.config(text=str(e), fg="red")
        return None, None, None, None

def train_create_model(message_label=None):
    global data_cleaned_and_split
    try:
        # Check if data has been cleaned and split
        if not data_cleaned_and_split:
            messagebox.showerror("Error", "Please clean and split the data first!")
            return None, None, None

        # Train the model
        X_train, X_test, y_train, y_test = clean_split_data()
        # model = LinearRegression()
        model = DecisionTreeRegressor(random_state=42)
        model.fit(X_train, y_train)
        print("Model trained successfully!")
        if message_label is not None:
            messagebox.showinfo("Model Training", "Model trained successfully!")
            message_label.config(text="Model trained successfully!", fg="green")
        return model, X_train, y_train
    except Exception as e:
        if message_label is not None:
            messagebox.showerror("Model Training", f"Error during model training: {str(e)}")
            message_label.config(text=str(e), fg="red")
        print("Error during model training:", e)
        return None, None, None

def predict_outcome(message_label=None):
    try:
        # Train and create the model

        model, X_train, y_train = train_create_model()
        print("Type of model:", type(model))
        # Get the input year
        valid_input = False
        while not valid_input:
            year = prediction_year.get()
            if not year.isdigit() or int(year) < 2020 or int(year) > 3000:
                if messagebox.askokcancel("Invalid Input", "Please enter a valid year between 2021 and 3000."):
                    # user clicked "OK"
                    continue
                else:
                    # user clicked "Cancel"
                    break
            else:
                valid_input = True

                # Close the message dialog box

        if valid_input:
            year = int(year)
            print("Year:", year)
            # Close the message dialog

        # Predict the outcome
        print(model.predict([[year]]))
        outcome = model.predict([[year]])
        print("outcome:", outcome)
        prediction_result.set(outcome[0])
        # Update the prediction result label
        prediction_result_label.config(text="Predicted outcome: {:.2f}".format(outcome[0]))
        print("Model:", model)
        print("Outcome: ", outcome)
        return model, outcome
    except Exception as e:
        # Check if a message label was provided
        if message_label is not None:
            message_label.config(text=str(e), fg="red")
def generate_report():
    model, outcome = predict_outcome()
    X_train, X_test, y_train, y_test = clean_split_data()
    # Create a new PDF object
    pdf_file = io.BytesIO()

    canvas_obj = canvas.Canvas(pdf_file, pagesize=letter)
    # Set the font and font size
    # Set the font and font size
    canvas_obj.setFont('Helvetica', 12)

    # Add a title to the PDF
    canvas_obj.drawCentredString(300, 750, "GENERAL REPORT")

    # Add some text to the PDF
    canvas_obj.drawCentredString(50, 730, "PART ONE:")
    canvas_obj.drawString(50, 700, "Model Information:")

    # Add performance metrics to the PDF
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mse_train = np.mean((y_train - y_pred_train) ** 2)
    mse_test = np.mean((y_test - y_pred_test) ** 2)
    r2_train = model.score(X_train, y_train)
    r2_test = model.score(X_test, y_test)

    # Add model performance metrics to the PDF
    canvas_obj.drawString(50, 680, "Mean Squared Error (Train): {:.2f}".format(mse_train))
    canvas_obj.drawString(50, 660, "Mean Squared Error (Test): {:.2f}".format(mse_test))
    canvas_obj.drawString(50, 640, "R-squared (Train): {:.2f}".format(r2_train))
    canvas_obj.drawString(50, 620, "R-squared (Test): {:.2f}".format(r2_test))
    # canvas_obj.drawString(50, 600, "Feature importances: {}".format(model.feature_importances_))
    # Add model coefficients and intercept to the PDF
    # canvas_obj.drawString(50, 580, "Model coefficients: {}".format(model.coef_))
    # canvas_obj.drawString(50, 560, "Model intercept: {}".format(model.intercept_))
    # Calculate regression performance metrics
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    evs_test = explained_variance_score(y_test, y_pred_test)

    canvas_obj.drawString(50, 580, "Mean Absolute Error (Test): {:.2f}".format(mae_test))
    canvas_obj.drawString(50, 560, "Root Mean Squared Error (Test): {:.2f}".format(rmse_test))
    # Add the predicted outcome to the PDF
    canvas_obj.drawString(50, 500, "Prediction Outcome:")
    canvas_obj.drawString(50, 480, "Input Year: {}".format(prediction_year.get()))
    canvas_obj.drawString(50, 460, "Predicted Outcome: {:.2f}".format(outcome[0]))

    # Add a bar plot of the predicted vs. actual values to the PDF
    canvas_obj.drawString(50, 430, "Data Visualization")
    canvas_obj.drawString(50, 410, "Bar Plot for Actual and Predicted Values")
    predicted = model.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': predicted})
    df.plot(kind='bar', figsize=(10, 5))
    plt.title('Actual vs Predicted')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Save the plot to a BytesIO object instead of a file
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()

    # Use the BytesIO object to create an ImageReader and draw the image onto the canvas
    buffer.seek(0)
    image = ImageReader(buffer)
    canvas_obj.drawImage(image, 50, 100, 400, 250)

    # Add some text to the PDF
    canvas_obj.showPage()
    canvas_obj.setFont('Helvetica', 12)
    canvas_obj.drawString(300, 750, "PART TWO:")

    # Create a data visualization and save it to a PNG file
    plt.clf()
    plt.plot(X_test[:, 0], y_test, label='True values')
    plt.plot(X_test[:, 0], y_pred_test, label='Predicted values')
    plt.xlabel('Year')
    plt.ylabel('Outcome')
    plt.title('True vs Predicted Outcome')
    plt.legend()
    plt.savefig('D:/report/line_graph.png', format='png')

    # Add the PNG file to the PDF
    canvas_obj.drawImage('line_graph.png', 50, 350, 500, 250)
    # Create a scatter plot and save it to a PNG file
    plt.clf()
    plt.scatter(X_train[:, 0], y_train, label='Training data')
    plt.scatter(X_test[:, 0], y_test, label='Testing data')
    plt.xlabel('Year')
    plt.ylabel('Outcome')
    plt.title('Scatter plot of training and testing data')
    plt.legend()
    plt.savefig('D:/report/scatter_plot.png', format='png')

    # Add the scatter plot PNG file to the PDF
    canvas_obj.drawImage('scatter_plot.png', 50, 100, 400, 250)

    # Save the PDF to a file
    canvas_obj.save()

    # Return the PDF file
    pdf_file.seek(0)
    print("file:", pdf_file)
    return pdf_file


def download_report():
    pdf_file = generate_report()
    print("pdf_file:", pdf_file)
    root = Tk()
    root.withdraw()
    file_path = filedialog.askdirectory(title="Select directory to save the report")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"report_{current_time}.pdf"
    with open(os.path.join(file_path, file_name), 'wb') as f:
        f.write(pdf_file.getbuffer())
    print("Report saved successfully at " + os.path.join(file_path, file_name))


# def reset_fields():
    file_path.set("")
    prediction_year.set("")
    prediction_result.set("")
    prediction_result_label.config(text="")  # set the text to an empty string

    # Create labels, buttons, and entry fields

root = Tk()
root.title("GDP PREDICTION")
root.geometry("700x500")
file_path = StringVar()
prediction_year = StringVar()
prediction_result = StringVar()

# Create the GUI window

Label(root, text="Upload Dataset:", font=("Arial", 16)).grid(row=2, column=0, padx=10, pady=10, sticky="E")
Entry(root, textvariable=file_path, width=40, font=("Arial", 14)).grid(row=2, column=1, columnspan=2, padx=10, pady=10, sticky="W")
Label(root, text="", font=("Arial", 16)).grid(row=2, column=0)

Button(root, text="Browse", command=browse_files, font=("Arial", 14)).grid(row=3, column=1, padx=10, pady=10)

Label(root, text="Clean and Split Data:", font=("Arial", 16)).grid(row=4, column=0, padx=10, pady=10, sticky="E")
Button(root, text="Clean and Split Data", command=clean_split_data, font=("Arial", 14)).grid(row=4, column= 1, padx=10, pady=10)

Button(root, text="Train Model", command=train_create_model, font=("Arial", 14)).grid(row=5, column=1, padx=10, pady=10)

Label(root, text="Input Year:", font=("Arial", 16)).grid(row=6, column=0, padx=10, pady=10, sticky="E")
Entry(root, textvariable=prediction_year, width=25, font=("Arial", 14)).grid(row=6, column=1, columnspan=2, padx=10, pady=10, sticky="W")

Label(root, text="Predict Outcome:", font=("Arial", 16)).grid(row=7, column=0, padx=10, pady=10, sticky="E")
Button(root, text="Predict", command=predict_outcome, font=("Arial", 14)).grid(row=7, column=1, padx=10, pady=10)
prediction_result_label = Label(root, text="Prediction Result", font=("Arial", 16))
prediction_result_label.grid(row=8, column=0, columnspan=3, padx=10, pady=10)

Button(root, text="Download Result in PDF", command=download_report, font=("Arial", 16)).grid(row=9, column=1, pady=10)
# Button(root, text="RESET", command=reset_fields, font=("Arial", 16)).grid(row=10, column=1, pady=10)
# center the window on the screen
root.update_idletasks()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width // 2) - (root.winfo_width() // 2)
y = (screen_height // 2) - (root.winfo_height() // 2)
root.geometry("+{}+{}".format(x, y))
root.mainloop()
