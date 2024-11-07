import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, url_for
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib
import folium

app = Flask(__name__)

# Load dataset and preprocess
train_file = 'static/data/Traffic.csv'
df = pd.read_csv(train_file)

# Time conversion and label encoding
df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.hour + pd.to_datetime(df['Time'], format='%I:%M:%S %p').dt.minute / 60
df['Date'] = df['Date'].astype(int)  # Convert Date to integer
day_encoder = LabelEncoder()
traffic_encoder = LabelEncoder()
df['Day of the week'] = day_encoder.fit_transform(df['Day of the week'])
df['Traffic Situation'] = traffic_encoder.fit_transform(df['Traffic Situation'])

# Split data for training and testing
X = df[['Time', 'Date', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount', 'Total', 'Day of the week']]
y = df['Traffic Situation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=50),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(kernel='rbf'),
    'Knearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Neural Network': MLPClassifier(activation='relu', hidden_layer_sizes=(100, 50), max_iter=1000)
}
model_accuracies = {}
model_dir = 'static/models'
os.makedirs(model_dir, exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracies[name] = accuracy_score(y_test, y_pred)
    joblib.dump(model, f'{model_dir}/{name.replace(" ", "_")}.pkl')

# Generate plots with clearer labels
def generate_plots():
    plot_dir = 'static/images'
    os.makedirs(plot_dir, exist_ok=True)

    df['Traffic Situation Label'] = traffic_encoder.inverse_transform(df['Traffic Situation'])
    df['Day of the Week Label'] = day_encoder.inverse_transform(df['Day of the week'])

    sns.set_theme(style="whitegrid")

    # 1. Traffic Situation Distribution
    plt.figure(figsize=(12, 12))
    sns.countplot(data=df, x='Traffic Situation Label')
    plt.savefig(os.path.join(plot_dir, 'traffic_situation_plot.png'))
    plt.close()

    # 2. Traffic Situation Over Time
    plt.figure(figsize=(12, 12))
    sns.histplot(data=df, x='Time', hue='Traffic Situation Label', bins=30)
    plt.savefig(os.path.join(plot_dir, 'traffic_time_plot.png'))
    plt.close()

    # 3. Traffic Situation by Day of the Week
    plt.figure(figsize=(12, 12))
    sns.countplot(data=df, x='Day of the Week Label', hue='Traffic Situation Label')
    plt.title("Traffic Situation for each day")
    plt.savefig(os.path.join(plot_dir, 'traffic_day_plot.png'))
    plt.close()

    # 4. Vehicle Type Distribution by Traffic Situation
    vehicle_columns = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']

    for vehicle in vehicle_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x=vehicle, hue='Traffic Situation Label', kde=True, multiple='stack')
        plt.title(f'{vehicle} Distribution by Traffic Situation')
        plt.savefig(os.path.join(plot_dir, f'{vehicle.lower()}_distribution_plot.png'))
        plt.close()

    # Plot model accuracies
    plot_model_accuracies(model_accuracies)

def plot_model_accuracies(model_accuracies):
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()))
    plt.title('Model Accuracies')
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/images/model_accuracies_plot.png')  # Save image to directory
    plt.close()

def create_map(prediction):
    color_map = {
        0: 'red',  
        1: 'orange',   
        2: 'green', 
        3: 'blue'    
    }
    color = color_map.get(prediction, 'white')  # Default to blue if not found

    # Coordinates for the route
    start_point = (10.012377, 105.732439)  # Start point
    end_point = (10.020460, 105.741596)    # End point
    nguyen_van_cu_route = [
        start_point,
        (10.014000, 105.734000),  # Intermediate point 1
        (10.016500, 105.737000),  # Intermediate point 2
        (10.018500, 105.739500),  # Intermediate point 3
        end_point
    ]

    # Create the map
    m = folium.Map(location=start_point, zoom_start=15)
    folium.Marker(location=start_point, popup='Start', icon=folium.Icon(color='green')).add_to(m)
    folium.Marker(location=end_point, popup='End', icon=folium.Icon(color='red')).add_to(m)

    # Draw the route with color based on prediction
    folium.PolyLine(locations=nguyen_van_cu_route, color=color, weight=5).add_to(m)

    # Save the map to HTML file
    map_file = 'static/traffic_map.html'
    m.save(map_file)
    return map_file

@app.route('/', methods=['GET', 'POST'])
def index():
    generate_plots()
    plots = {
        'Traffic Situation Distribution': url_for('static', filename='images/traffic_situation_plot.png'),
        'Traffic Situation Over Time': url_for('static', filename='images/traffic_time_plot.png'),
        'Traffic Situation by Day of the Week': url_for('static', filename='images/traffic_day_plot.png'),
        'Car Distribution': url_for('static', filename='images/carcount_distribution_plot.png'),
        'Bike Distribution': url_for('static', filename='images/bikecount_distribution_plot.png'),
        'Bus Distribution': url_for('static', filename='images/buscount_distribution_plot.png'),
        'Truck Distribution': url_for('static', filename='images/truckcount_distribution_plot.png'),
        'Model Accuracies': url_for('static', filename='images/model_accuracies_plot.png')  # Add model accuracies plot
    }
    prediction, selected_model, traffic_map = None, None, 'static/traffic_map.html'

    if request.method == 'POST':
        selected_model = request.form['model']
        model = joblib.load(f'static/models/{selected_model.replace(" ", "_")}.pkl')

        try:
            # Convert time input to the correct format
            time_input = request.form['Time']
            time_encoded = pd.to_datetime(time_input, format='%I:%M:%S %p').hour + pd.to_datetime(time_input, format='%I:%M:%S %p').minute / 60
            
            input_data = [
                time_encoded,
                int(request.form['Date']),  # Add this line to include Date
                int(request.form['CarCount']),
                int(request.form['BikeCount']),
                int(request.form['BusCount']),
                int(request.form['TruckCount']),
                int(request.form['Total']),
                day_encoder.transform([request.form['Day of the week']])[0]
            ]
            pred_label = model.predict([input_data])[0]
            prediction = traffic_encoder.inverse_transform([pred_label])[0]

            # Create traffic map based on prediction
            traffic_map = create_map(pred_label)
        except ValueError as e:
            prediction = f"Error processing input: {str(e)}"

    return render_template(
        'index.html',
        plots=plots,
        models=model_accuracies.keys(),
        model_accuracies=model_accuracies,
        prediction=prediction,
        selected_model=selected_model,
        traffic_map=traffic_map
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
