
# 🤖 Gmail CLI AI Utility

A Python command-line tool that integrates Gmail with AI capabilities using the Gmail API and OpenAI through LangChain.

It allows users to fetch and analyze Gmail messages directly from the terminal while leveraging AI to summarize and understand email content.

Built with modular classes for easy extension and future AI-powered automation.

---

## 🔧 Features

| Feature | Status |
|--------|--------|
| Fetch last N emails | ✔ |
| Fetch by sender email | ✔ |
| AI summarize email content | ✔ |
| AI analyze email intent | ✔ |
| OAuth-based Gmail authentication | ✔ |

---

## 🛠 Requirements

- Python 3.8+
- Google API Credentials (`credentials.json`)
- Gmail API enabled in Google Cloud Console
- AI API key (depending on provider)

Installed dependencies:
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib openai



## 🚀 Usage

Run the CLI:


python main.py


Menu interface provides options to:


1. Fetch last N mails
2. Fetch mails by email address
3. Summarize an email using AI
4. Analyze email content
5. Exit



The AI module processes selected email messages and generates intelligent summaries or insights.

---

## 🔐 Security Notice

Add these to `.gitignore` (already recommended):



credentials.json
token.json
.env
**pycache**/



Never upload Gmail credentials or API keys publicly.

---

## 📄 Project Structure



gmail-cli-ai/
├── main.py
├── auth.py
├── reader.py
├── ai_processor.py
├── utils.py
├── config.py
├── credentials.json   (ignored)
├── token.json         (ignored)



---

## ⭐ Contributions & Ideas

PRs and feature suggestions are welcome.  
If you use it — consider giving the repo a ⭐.

---

## 👨‍💻 Author

**Bishwayan Ghosh**

GitHub:  
https://github.com/GhoshBishwayan

✅ Steps:

1. Open your **gmail-cli-ai repo**
2. Edit **README.md**
3. **Delete old content**
4. **Paste this**
5. Commit

---


