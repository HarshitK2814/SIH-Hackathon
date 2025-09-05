# 🚜 RoboCrop - AI-Powered Agricultural Disease Detection

**Developed for the Smart India Hackathon 2025.**

RoboCrop is a smart farming assistant that uses a state-of-the-art Deep Learning model to detect diseases in crops with **96.45% accuracy**. This tool empowers farmers to diagnose plant health issues instantly using just a smartphone photo, enabling early treatment and protecting crop yields.

![RoboCrop UI](https://github.com/HarshitK2814/SIH-Hackathon/blob/master/images/Dashboard_RoboCrop.jpg)


---

## 🌱 Key Features

- **High-Accuracy AI Model**: A Roboust Convolutional Neural Network (CNN) trained on a diverse dataset to identify multiple diseases across various crops like tomatoes, potatoes, and peppers.
- **Instant Diagnosis**: Get results, confidence scores, and actionable treatment recommendations in seconds.
- **User-Friendly Interface**: A clean, intuitive, and farmer-friendly web application built with Streamlit, designed for ease of use in the field.
- **Scalable Architecture**: A Roboust FastAPI backend serves the PyTorch model, ensuring high performance and the ability to handle multiple requests.
- **Dockerized for Portability**: The entire application is containerized using Docker and Docker Compose, allowing for easy, consistent, and reliable deployment on any platform.

---

## 🚀 Tech Stack

- **Backend**: Python, FastAPI, Uvicorn
- **Machine Learning**: PyTorch, Torchvision, OpenCV
- **Frontend**: Streamlit
- **Code Management**: Git, GitHub

---

## 🔧 Getting Started

Follow these instructions to get a local copy of the project up and running for development and testing purposes.

### Prerequisites

- [Git](https://git-scm.com/downloads) installed on your system.

### Installation & Setup

**1. Clone the repository:**
```bash
git clone https://github.com/HarshitK2814/SIH-Hackathon.git
cd SIH-Hackathon
```

**2. Build and run with Docker Compose:**
This single command will build the Docker images for the backend and frontend, and start both services.
```bash
docker-compose up --build
```
*(The initial build may take a few minutes to download the base images and install dependencies.)*

**3. Access the application:**
Once the containers are up and running, you can access the services at the following URLs:
- **🌱 RoboCrop Frontend UI**: `http://localhost:8501`
- **⚙️ Backend API Docs**: `http://localhost:8000/docs`

---

## 📁 Project Structure

The repository is organized to separate concerns and maintain a clean, scalable architecture:

```
.
├── backend/                # Contains all FastAPI and AI model logic
│   ├
│   ├── main.py            # FastAPI application entrypoint
│   ├── inference.py       # Model prediction logic
│   ├── model.py           # CNN model architecture
│   └── requirements.txt
│
├── frontend/              # Contains all Streamlit UI code
│   ├
│   ├── app.py            # Main Streamlit application
│   └── requirements.txt
│
├
└── .gitignore            # Specifies files to be ignored by Git
```

---

## 🎯 Project Goal & Impact

The primary goal of RoboCrop is to bridge the technology gap for farmers by providing an accessible, affordable, and accurate tool for crop health management. By enabling early disease detection, this project aims to:

- **Reduce Crop Loss**: Help farmers save a significant portion of their yield that would otherwise be lost to disease.
- **Optimize Pesticide Use**: Promote targeted treatment instead of broad, untargeted spraying, saving costs and reducing environmental impact.
- **Empower Farmers**: Give farmers the data they need to make informed decisions, improving their livelihoods and contributing to food security.

This project is our contribution to building a more sustainable and technologically advanced agricultural ecosystem.

---

## 📊 Model Performance

- **Accuracy**: 96.45%
- **Model Type**: Convolutional Neural Network (CNN)
- **Framework**: PyTorch
- **Training Dataset**: Diverse crop disease dataset covering multiple plant species

---

## 🙏 Acknowledgments

- Smart India Hackathon 2025 for providing the platform to develop this solution
- The open-source community for the amazing tools and libraries
- Farmers and agricultural experts who inspired this project

---

**Made with 🌾 for a better agricultural future**
