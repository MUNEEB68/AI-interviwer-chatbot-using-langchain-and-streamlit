# AI-interviwer-chatbot-using-langchain-and-streamlit
Here's a more detailed README description that includes the technical aspects of your project:

---

# AI Job Application Assistant

This repository contains the codebase for an AI-powered Job Application Assistant, designed to streamline the process of applying for jobs. The assistant can store job positions, parse CVs, evaluate candidates' skills and experience, and even conduct coding interviews with real-time feedback.

## Features

### 1. **Job Position Management**
   - **Storage & Retrieval**: Store and retrieve job positions with ease, making it simple for candidates to find relevant job openings.
   - **Uploaded the file that contains all the availabe job roles
   - **Job Matching**: Automatically match candidates to the most suitable positions based on their experience and skills.

### 2. **CV Parsing and Analysis**
   - **CV Upload**: Candidates can upload their CVs in various formats (PDF, DOCX, etc.).
   - **Skill & Experience Extraction**: Using LangChain and FAISS (Facebook AI Similarity Search), the assistant extracts relevant skills, experiences, and qualifications from CVs to match them against job descriptions.
   - **Semantic Search**: FAISS is leveraged for efficient semantic search, enabling the bot to match job descriptions with candidate profiles based on context rather than just keywords.

### 3. **Eligibility Check and Skill Evaluation**
   - **Initial Screening**: The assistant checks the eligibility of candidates based on predefined criteria, ensuring that only qualified candidates proceed.
   - **Custom Interview Questions**: The bot generates tailored interview questions based on the candidate's CV, dynamically adjusting the difficulty and focus of the questions.
   - **Real-time Follow-up Questions**: The assistant responds to candidates' answers with follow-up questions, enhancing the assessment process by diving deeper into specific areas of expertise.

### 4. **Coding Challenges and Evaluation**
   - **Dynamic Coding Questions**: Candidates are presented with coding challenges that match the job requirements.
   - **Automated Code Evaluation**: The assistant uses a custom code evaluator to assess the candidates' submissions based on logic, correctness, time complexity, and other factors. Feedback is provided in real-time.
   - **Reattempt Option**: Candidates can reattempt questions, with the assistant offering guidance on where they need to improve.

### 5. **Interactive and Personalized Experience**
   - **Candidate Interaction**: The bot interacts with candidates by using their names and engaging in a conversational manner, providing a personalized experience.
   - **Voice and Text Integration**: The assistant supports chat-to-text and text-to-voice functionality, making it accessible for a wide range of users.
   - **Email Notifications**: Automated email notifications keep candidates informed about their application status and provide detailed feedback after interviews.

## Getting Started

### Prerequisites
- Python 3.x
- OpenAI API Key
- FAISS
- LangChain

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/ai-job-application-assistant.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the main application:
   ```bash
   python main.py
   ```
2. Follow the on-screen instructions to interact with the assistant.

## Contributing
We welcome contributions to enhance the functionality and usability of the AI Job Application Assistant. Please feel free to submit pull requests or open issues for any bugs or feature requests.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README description provides a clear and comprehensive overview of your project, emphasizing its features and the technologies used to implement them.
