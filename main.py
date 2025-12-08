import os

# Base folder
base_folder = "Full_ML_Roadmap"
os.makedirs(base_folder, exist_ok=True)

# Folder structure with topic names
folders = [
    "01_Unsupervised_Learning",
    "02_ANN",
    "03_CNN",
    "04_NLP",
    "05_TimeSeries",
    "06_Streamlit_Apps"
]

# Datasets links for reference
datasets_links = {
    "01_Unsupervised_Learning": [
        ("Mall_Customers.csv", "https://raw.githubusercontent.com/marcopeix/cluster-analysis/master/Mall_Customers.csv"),
        ("Credit_Card_Customers.csv", "https://raw.githubusercontent.com/plotly/datasets/master/cc_data.csv")
    ],
    "02_ANN": [
        ("MNIST", "Use `mnist.load_data()` in code"),
        ("Fashion_MNIST", "Use `fashion_mnist.load_data()` in code"),
        ("Churn_Modelling.csv", "https://raw.githubusercontent.com/sharmaroshan/Customer-Churn-Prediction/master/Churn_Modelling.csv")
    ],
    "03_CNN": [
        ("CIFAR-10", "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"),
        ("Cats_Dogs", "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip")
    ],
    "04_NLP": [
        ("IMDB", "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"),
        ("Twitter_Sentiment.csv", "https://raw.githubusercontent.com/laugustyniak/bert-sentiment-analysis/master/data/Tweets.csv")
    ],
    "05_TimeSeries": [
        ("AirPassengers.csv", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"),
        ("DailyMinTemp.csv", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")
    ],
    "06_Streamlit_Apps": []
}

# Create folders and README.md
for folder in folders:
    folder_path = os.path.join(base_folder, folder)
    os.makedirs(folder_path, exist_ok=True)
    readme_path = os.path.join(folder_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# {folder.replace('_', ' ')}\n")
        f.write(f"This folder is for {folder.replace('_', ' ')}.\n\n")
        if datasets_links[folder]:
            f.write("## Datasets Links:\n")
            for name, link in datasets_links[folder]:
                f.write(f"- {name}: {link}\n")

# Generate Roadmap.md
roadmap_md_path = os.path.join(base_folder, "Roadmap.md")
with open(roadmap_md_path, "w") as f:
    f.write("# 60-Day ML/DL Roadmap (Streamlit Deployment)\n")
    f.write("Deep Learning • Unsupervised • ANN • CNN • NLP • Time Series • Streamlit Deployment\n\n")
    f.write("## Month 1 — Deep Learning Foundations\n")
    f.write("- Week 1: Unsupervised Learning (K-Means, PCA, DBSCAN, Anomaly Detection)\n")
    f.write("- Week 2: ANN (Dense Networks, Digit Classifier)\n")
    f.write("- Week 3: CNN (CIFAR-10, Cats vs Dogs, Transfer Learning)\n")
    f.write("- Week 4: Advanced CV (YOLO, Lane/Garbage/Helmet Detection)\n\n")
    f.write("## Month 2 — NLP + Time Series + Streamlit Deployment\n")
    f.write("- Week 5: NLP Foundations (Tokenization, TF-IDF, Sentiment/Spam Classifier)\n")
    f.write("- Week 6: Advanced NLP (Word2Vec, LSTM, BERT)\n")
    f.write("- Week 7: Time Series (ARIMA, SARIMA, Prophet, Forecasting Projects)\n")
    f.write("- Week 8: Streamlit Deployment (Interactive ML/DL dashboards)\n\n")
    f.write("## Datasets included:\n")
    for folder in folders:
        if datasets_links[folder]:
            for name, link in datasets_links[folder]:
                f.write(f"- {name}: {link}\n")

print("✔ All folders, README.md, and Roadmap.md created successfully!")
