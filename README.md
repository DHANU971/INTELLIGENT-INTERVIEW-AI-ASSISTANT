# 🎙️ Intelligent Interview AI Assistant

An AI-powered mock interview platform that simulates real technical interviews using **Groq's LLaMA 3.1** model, offering real-time face analysis, resume parsing, and detailed performance feedback.

---

## ✨ Features

- **AI-Driven Interviews** – Conducts adaptive technical interviews powered by LLaMA 3.1 via Groq
- **Resume-Aware Questions** – Upload your PDF resume and the AI tailors questions to your experience
- **Custom Question Mode** – Paste your own question list for the AI to ask one-by-one
- **Difficulty Levels** – Easy, Medium, and Hard modes to match your preparation level
- **Real-Time Face Analysis** – Uses OpenCV to detect presence and estimate confidence/nervousness
- **Performance Feedback** – Generates a structured JSON report with score, strengths, and improvement areas
- **Session Memory** – Full conversation history maintained per session using LangChain

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| AI / LLM | Groq API (LLaMA 3.1 8B Instant) |
| LLM Framework | LangChain (Core, Groq, Community) |
| Face Analysis | OpenCV (Haar Cascade) |
| Resume Parsing | PyPDF2 |
| Environment | python-dotenv |

---

## 📁 Project Structure

```
INTELLIGENT INTERVIEW AI ASSISTANT/
├── app.py               # Main Flask application
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (not committed)
├── templates/
│   └── index.html       # Frontend UI
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone / Navigate to the Project
```bash
cd "INTELLIGENT INTERVIEW AI ASSISTANT"
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
SECRET_KEY=your_flask_secret_key_here
```

> Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the Application
```bash
python app.py
```

Open your browser and go to: **http://localhost:5000**

---

## 🚀 How to Use

1. **Configure** – Select a job role, difficulty level, topic focus, and interview mode (Chat/Video)
2. **Upload Resume** *(optional)* – Upload a PDF to get resume-specific questions
3. **Start Interview** – The AI asks one question at a time and adapts based on your answers
4. **Finish** – Click "Finish" to receive a detailed feedback report with your score and hiring recommendation

---

## 📊 Feedback Report

At the end of each session, the AI generates:
- **Score** (0–100)
- **Feedback Summary**
- **Strengths**
- **Areas for Improvement**
- **Hiring Recommendation** (Hired / Not Hired)

---

## 📦 Dependencies

```
flask
langchain
langchain-core
langchain-groq
langchain-community
python-dotenv
PyPDF2
opencv-python
numpy
```

---

## 📄 License

This project was built as a college project for educational purposes.
