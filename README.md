
---

# ⚙️ Automated Resume Screening with NLP

An intelligent resume screening tool that leverages Natural Language Processing (NLP) to extract key information, match candidate profiles with job descriptions, and assist in shortlisting top talent.

---

## 🚀 Features

- 📄 Extracts text from PDF resumes
- 🧠 Performs semantic matching with job descriptions
- ✅ Identifies relevant skills, experiences, and ranks candidates
- 🖥️ Streamlit-based interactive UI for easy review and shortlisting
- 🗃️ Stores data locally using SQLite via SQLAlchemy

---

## 🛠️ Tech Stack

- **Python**: Core language
- **spaCy**: NLP engine for text extraction and linguistic analysis
- **Hugging Face Transformers**: Semantic similarity and deep language modeling
- **Streamlit**: Web-based interactive UI
- **SQLAlchemy + SQLite**: Lightweight database for storing candidate info

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Hash-Kalyan/resume-screening-nlp.git
cd resume-screening-nlp
pip install -r requirements.txt
python -m spacy download en_core_web_sm
````

---

## ▶️ Running the Application

Start the Streamlit app:

```bash
streamlit run app/main.py
```

Once running, access the app in your browser at: [http://localhost:8501](http://localhost:8501)



## 🤝 Contributing

Contributions are welcome! Feel free to submit issues or pull requests to enhance the functionality or UI.

---

## 📬 Contact

Built by [Hasvanth Kalyan Gundu](https://www.linkedin.com/in/hasvanth-kalyan-g-13538a148)
📧 [hasvanthkalyan9@gmail.com](mailto:hasvanthkalyan9@gmail.com)

---

