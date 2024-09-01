import os
import openai
import requests
import json
import time
import logging
from datetime import datetime
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import smtplib
# Securely load the API key
openai_api_key = #add your openai api key
openai.api_key = openai_api_key

# Read PDF file using PdfReader
# File path to the PDF
pdf_file_path = r"C:\Users\LENOVO\Desktop\jobs_company.pdf"

# Check if the file exists
if os.path.exists(pdf_file_path):
    print("File exists, proceeding to read the PDF.")
    
    # Read PDF file using PdfReader
    pdf_reader = PdfReader(pdf_file_path)

    # Extract text from each page
    text1 = ""
    for page in pdf_reader.pages:
        text1 += page.extract_text()

else:
    print("File does not exist. Please check the file path and name.")

# Ensure that the job descriptions are correctly embedded
system_message = f"""
You are a friendly interviewer assistant for my company for IT roles. You have to follow the following steps I give you one by one.
Here are the jobs available in my company with descriptions:
{text1} 
"""
# Specify the model to be used
model = "gpt-3.5-turbo"

class InterviewerAssistant:
    def __init__(self):
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
            st.session_state.conversation_history.append({"role": "system", "content": system_message})
        self.model = model

    def get_response(self, user_input, temperature1=0.7):
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=st.session_state.conversation_history,
                temperature=temperature1,
                max_tokens=1000,
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return "Sorry, I couldn't process that request."

    def add_message_to_conversation(self, role, content):
        st.session_state.conversation_history.append({"role": role, "content": content})

    def get_skills_from_pdf(self, pdf_file):
        pdfreader = PdfReader(pdf_file)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=200,
            chunk_overlap=50,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        document_search = FAISS.from_texts(texts, embeddings)

        llm = OpenAI(openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")

        query = "What are the skills of the candidate? Please return them as a comma-separated list. Find the technical skills in the skills portion of resume and give skills according to that. Do not use the words 'or' and '/' in your response. Just give the name of the skills and don’t use any general words such as technical skills or database management, etc./And no repetition of same skill please"
        docs = document_search.similarity_search(query)
        skills = chain.run(input_documents=docs, question=query)
        skills = skills.split(",")
        skill_list = [skill.strip() for skill in skills if skill.strip()]
        return skill_list

    def get_any_data_from_pdf(self, pdf_file, message):
        pdfreader = PdfReader(pdf_file)
        raw_text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        document_search = FAISS.from_texts(texts, embeddings)

        llm = OpenAI(openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")

        docs = document_search.similarity_search(message)
        result = chain.run(input_documents=docs, question=message)
        return result

    def get_skills_from_jobdescription(self, raw_text):
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        document_search = FAISS.from_texts(texts, embeddings)

        llm = OpenAI(openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        message = """From the raw text provided, extract a list of skills that are mentioned in the text.
        I only want the name of the skills e.g. HTML, CSS, etc., not extra sentences like 'experience in front-end technologies'.
        The output should not contain any general terms, e.g., 'Database management system' should have the exact name of the tool such as MySQL, etc.
        The skills should be a string separated by commas (,).
         """
        docs = document_search.similarity_search(message)
        result = chain.run(input_documents=docs, question=message)
        skills = result.split(",")
        skill_list = [skill.strip() for skill in skills if skill.strip()]
        return skill_list
    def get_any_data_fromtext(self,raw_text,messages):
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=350, chunk_overlap=50)
        texts = text_splitter.split_text(raw_text)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        document_search = FAISS.from_texts(texts, embeddings)

        llm = OpenAI(openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff")
        message = messages
        docs = document_search.similarity_search(message)
        result = chain.run(input_documents=docs, question=message)
        return result

    def match_job_role(self, candidate_skills, job_description):
        required_skills = self.get_skills_from_jobdescription(job_description)
        print("Required_skills:",required_skills)
        matched_skills = set(candidate_skills) & set(required_skills)
        match_percentage = (len(matched_skills) / len(required_skills)) * 100
        return match_percentage

    def check_if_file_is_pdf(self, file):
        if file is not None:
            if file.type == "application/pdf":
                return f"file_name:{file.name}"
            else:
                return "Not a PDF file. Please upload again."
        return "No file uploaded."
class Grader:
    def __init__(self):
        system_message1 = """You are an expert programmer and grader. You will be provided with a conversation history regarding the interaction 
        between an interviewer and a candidate for an IT-related role. You will have to grade it out of 10. The interaction could include either 
        an interview or coding questions and answers by the candidate."""
        if "conversation_history2" not in st.session_state:
            st.session_state.conversation_history2 = []
            st.session_state.conversation_history2.append({"role": "system", "content": system_message1})
        self.model = model

    def extract_interview_data(self, Assistant,conversation_history):
        message = f"Extract all the interview questions asked by the interviewer and the answers given by the candidate from the conversation history {conversation_history}/Please give the whole question and the whole answer given./There are going to be 3 questions and 3 answers/"
        response = Assistant.get_response(message)
        st.session_state.grade_conversation = True
        return response  # Return the extracted interview data
    

    def extract_coding_data(self, Assistant,conversation):
        message = f"Extract all coding questions asked by the interviewer and the answers given by the candidat.from the conversation history {conversation}"
        response = Assistant.get_response(message)
        return response  # Return the extracted coding data

    def grade(self, Assistant, conversation_history):
        # Add the conversation history to the grader's conversation history
        st.session_state.conversation_history2.append({"role": "user", "content": conversation_history})
        
        # Ask the assistant to grade the interaction
        grading_prompt = """
        Based on the conversation history, evaluate the candidate's performance and provide a grade out of 10.
        Consider factors like relevance of answers, clarity, correctness, and completeness and detail etc
        Provide a brief explanation for the grade you give. / You have to directly adress the candidate
        as this message will be shown directly to the candidate./
        """
        st.session_state.conversation_history2.append({"role": "user", "content": grading_prompt})
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=st.session_state.conversation_history2,
                temperature=0.5,
                max_tokens=500,
            )
            return response['choices'][0]['message']['content']  # Return the grading response
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return "Sorry, I couldn't process the grading request."
class Final_Email_generator:
    def __init__(self):
        system_message =f"""
        You are an email generator for an IT company. You will be given two responses from assistants. 
        One will be about the interview result of a candidate, and one will be about the coding interview result. 
        You need to first find the scoreobtained/total for the interview result/
        Then you need to find the score obtained/total for the coding interview result/
        Then you will have to find the overall total score that is (score obtained in interview+score obtained in coding test)/total/
        If the score is greater than or equal to 60 percent, generate an email for selection for the next round; 
        if less than 60 percent, generate a rejection email.
        Make sure to include the overall score as part of the email/
        and lastly only return the main body of the email that starts with the name of the candidate{st.session_state.candidate_name}
        """
        if "Email_generation" not in st.session_state:
            st.session_state.Email_generation = [{"role": "system", "content": system_message}]
        self.model = model

    def generate_email(self, interview_result, coding_result):
        st.session_state.Email_generation.append({"role": "assistant", "content": f"Interview result: {interview_result}\nCoding result: {coding_result}"})
        
        # Construct the email generation prompt
        email_prompt = "Based on the interview and coding results provided,first calculate the total score and generate the appropriate email to the candidate./only return the body of the email"
        st.session_state.Email_generation.append({"role": "user", "content": email_prompt})
        
        try:
            # Generate the email using the OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=st.session_state.Email_generation,
                temperature=0.5,
                max_tokens=500,
            )
            return response['choices'][0]['message']['content']  # Return the generated email content
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return "Sorry, I couldn't process the email generation request."

class Code_Evaluator:
    def __init__(self):
        syestem_message10 = """
You are an expert code evaluator. Your task is to assess a candidate's answers to a series of coding questions. 
Each question is worth 5 marks. Evaluate the responses based on the following criteria:
- Logic and approach
- Correctness of the solution
- Time and space complexity
- Code readability and efficiency

Provide detailed feedback on each aspect and assign marks accordingly.
Each question carries 5 marks so final score will depend on number of questions asked e.g if 3 questions asked
total marks will be 15
"""
        self.model = model
        if  "Programming_questions_history" not in st.session_state:
            st.session_state.Programming_questions_history = []
            st.session_state.Programming_questions_history.append({"role": "system", "content": syestem_message10})
    def Evaluate_questions(self,convesation_history):
        convesation_history.append({"role": "user", "content":"Look at the whole conversation history and evaluate all the questions/most probably there will be 3 questions.and give final result marksobtained/totalmarks for all of the questions"})
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages= convesation_history,
                temperature=0.5,
                max_tokens=500,
            )
            return response['choices'][0]['message']['content']  # Return the grading response
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return "Sorry, I couldn't process the grading request."  
def handle_cv_upload(Assistant, placeholder):
    if "step1_execution_time" not in st.session_state:
        st.session_state.step1_execution_time = True
    if st.session_state.step1_execution_time:
      uploaded_file = st.file_uploader("Upload your CV", type="pdf")
      if uploaded_file is not None:
        Assistant.add_message_to_conversation("user", "Uploaded CV")
        Assistant.add_message_to_conversation("assistant", "Processing your CV...")
        placeholder.success(f"Uploaded: {uploaded_file.name}")
        
        # Extract skills
        skills = Assistant.get_skills_from_pdf(uploaded_file)
        st.session_state.candidate_skills = skills
        
        # Extract experience
        candidates_experience = Assistant.get_any_data_from_pdf(uploaded_file,"Get all the info related to the candidate's experience from the file uploaded. The info should consist of the candidate's experience, skills in which they are proficient, projects they have worked on, and educational details.")
        st.session_state.candidates_experience = candidates_experience
        user_email_adress =  Assistant.get_any_data_from_pdf(uploaded_file,"find the email adress of candidate and return it/ the response should contain only the email adress and nothing else")
        user_email_adress = user_email_adress.split()
        for item in user_email_adress:
            if "@" in item:
                st.session_state.user_email_adress = item
                break
        # Extract candidate name
        candidate_name = Assistant.get_any_data_from_pdf(uploaded_file,"Return the candidate name. Only the name, nothing else.")
        st.session_state.candidate_name = candidate_name

        # Display extracted data
        #placeholder.text(f"Candidate Name: {candidate_name}")
        #st.text("Extracted Skills:")
        #st.text(", ".join(skills))
        #st.text("Experience Details:")
        #st.text(candidates_experience)
        
        st.session_state.step1_execution_time = False
        # Set flag to show conversation history
        st.session_state.cv_uploaded = True
      else:
        placeholder.warning("Please upload a PDF file.")
def Handle_jobs_roleselection(Assistant):
    # Only initialize step2_executed if it hasn't been set before
    if "step2_executed" not in st.session_state:
        st.session_state.step2_executed = True
    if "inner_if_executed"  not in st.session_state:
        st.session_state.inner_if_executed = True


    # Only proceed if step2_executed is True
    if st.session_state.step2_executed:
        message = """
        The message should start with the wording 'Hope you are doing well.'
        Display the names of the available job roles in the company with serial numbers 1 to the total number of jobs in a friendly way.
        Don't ask the candidate for his CV and tell him to wait for 20 seconds before entering his selected role.
        """
        response = Assistant.get_response(message, temperature1=0.0)
        Assistant.add_message_to_conversation("assistant", response)
        st.text_area("Assistant", response, height=200, disabled=True)
        st.session_state.step2_executed = False
    if st.session_state.step2_executed == False:
      if st.session_state.inner_if_executed:  
        job_selection = st.text_input("Enter the serial number of the job you want to apply for:")
        if job_selection:
            try:
                selected_number = int(job_selection)  # Ensure input is an integer
                st.session_state.serial_number = selected_number

                with st.spinner('Processing... Please wait for 20 seconds'):
                    time.sleep(20)  # Simulate processing time

                Assistant.add_message_to_conversation("user", job_selection)
                st.success(f"You have selected the job role at serial number: {selected_number}")

                # Mark the job as selected
                st.session_state.job_selected = True

                # Now disable this step from running again
                st.session_state.step2_executed = False
                st.session_state.inner_if_executed =False

            except ValueError:
                st.error("Please enter a valid serial number.")
def Check_eligibility(Assistant, placeholder):
    if "Eligibility_check" not in st.session_state:
        st.session_state.Eligibility_check = True
    
    if st.session_state.Eligibility_check:
        try:
            # Fetch the job description and required skills using the selected serial number
            message = f"""
            Get the job description and skills required section of the job the user has selected.
            The user has selected the job at serial number: {st.session_state.serial_number}.
            """
            response = Assistant.get_response(message)
            print(response)
            Assistant.add_message_to_conversation("assistant", response)

            if not response.strip():
                st.error("Could not retrieve job details. Please try again.")
                return

            # Extract candidate skills from the session state
            candidate_skills = st.session_state.candidate_skills
            print(candidate_skills)
            if not candidate_skills:
                st.error("Could not extract candidate skills. Please check the PDF.")
                return

            # Calculate the match percentage using the existing function
            eligibility_percentage = Assistant.match_job_role(candidate_skills, response)
            print(eligibility_percentage)

            # Display eligibility result
            if eligibility_percentage >= 30:
                st.success("You are eligible for the job!")
                Assistant.add_message_to_conversation("user", "The candidate is eligible for this job")
            else:
                st.error("You are not eligible for the job! Thank you for applying.")
                st.stop()

        except Exception as e:
             #logging.error(f"An error occurred during eligibility check: {e}")
            st.error("An error occurred during eligibility check. Please try again.")
        
        # Mark eligibility check as completed
        st.session_state.Eligibility_check = False
        st.session_state.CV_verified = True
def Generate_interview_questions(i, Assistant, placeholder,key1):
    # Ensure that the candidate's information is added only once to the conversation history
    if "candidate_info" not in st.session_state:
        if "here is the info about the candidate" not in [msg["content"] for msg in st.session_state.conversation_history]:
            Assistant.add_message_to_conversation("user", f"here is the info about the candidate:{st.session_state.candidates_experience}")
        st.session_state.candidate_info = True

    if f"step{i}_question" not in st.session_state:
        st.session_state[f"step{i}_question"] = True

    if st.session_state[f"step{i}_question"]:
        previous_answer = st.session_state.get(f"user_input{i-1}", "N/A")
        prompt = f""" /First of all see the info about the candidate provided in the conversation history so you can generate better responses/
Given the candidate's previous answer: '{previous_answer}', if no previous answer then talk about info about candidate
Otherwise, provide feedback that is either positive or constructive, depending on whether the answer was strong or needed improvement.
/Next, generate a follow-up question that builds on the candidate's previous experience and skills.
Address the candidate by name ({st.session_state.candidate_name}) in a friendly and engaging manner/.
"""
        response = Assistant.get_response(prompt)
        Assistant.add_message_to_conversation("assistant", response)
        st.text_area(f"Question_number{i}", response, height=100, disabled=True)
        st.session_state[f"step{i}_question"] = False
    

    # Capture user input immediately after the question is generated
    user_response = st.text_input("Your response",key=key1)
    if user_response:
        Assistant.add_message_to_conversation("user", user_response)
        st.session_state[f"user_input{i}"] = user_response
        if key1 == "Question_1":
           st.session_state.First_Interview_quesrion_answered = True
        print(key1)
        if key1 == "Question_2":
             st.session_state.Second_interview_quesrion = True
        if key1 == "Question_3":
            st.session_state.third_interview_question = True
def Generate_coding_questions(Assistant, place_holder):
    if "coding_question_counter" not in st.session_state:
        st.session_state.coding_question_counter = 0

    # Dynamically generate keys for storing responses and controlling question flow
    i = st.session_state.coding_question_counter
    key2 = f"Coding_Question{i}"

    # Store the generated question for future reference
    if f"previous_question" not in st.session_state:
        st.session_state.previous_question = "No question asked yet"

    if f"Coding_step{i}_question" not in st.session_state:
        st.session_state[f"Coding_step{i}_question"] = True

    print(i)
    if st.session_state[f"Coding_step{i}_question"]:
        previous_question = st.session_state.previous_question    
        print(previous_question)
        prompt = f"""
        Given the last coding question: '{previous_question}\n':
        - Generate a new coding question that is a DSA problem and harder than the previous one./
        -If the last question was :"No question asked yet" then generate a simple (e.g., array traversal) question.
        /- If the last question was simple (e.g., array traversal), make this one medium difficulty (e.g., recursion, sorting with edge cases).
        - If the last question was medium difficulty, make this one hard (e.g., dynamic programming, graph traversal, advanced trees).

        Address the candidate by name ({st.session_state.candidate_name}) in a friendly and motivating manner and donot give any hints ever.
        """

        response = Assistant.get_response(prompt)
        st.session_state.Programming_questions_history.append({"role": "assistant", "content": response})
        Assistant.add_message_to_conversation("assistant", response)
        st.session_state.previous_question = response
        
        st.text_area(f"CodingQuestion_number{i}", response, height=100, disabled=True)
        
        # Store the current question in session state for future reference
        
        st.session_state[f"Coding_step{i}_question"] = False

    user_response2 = st.chat_input("Your response", key=key2)
    if user_response2:
        Assistant.add_message_to_conversation("user", user_response2)
        st.session_state.Programming_questions_history.append({"role": "user", "content": user_response2})
        st.session_state[f"Coding_user_input{i}"] = user_response2
        st.session_state.coding_question_counter += 1
        print(st.session_state.coding_question_counter)

         
        # Check if the coding test is done after 3 questions
        if st.session_state.coding_question_counter >= 3:
            st.session_state.coding_test_done = True
        
        if st.session_state.coding_question_counter <3:

          if st.button("Move forward"):
            st.write("")
         
def handle_interview_step(step, Assistant, placeholder):
    step_functions = {
        1: lambda: handle_cv_upload(Assistant, placeholder),
        2 : lambda: Handle_jobs_roleselection(Assistant),
        3 : lambda: Check_eligibility(Assistant,placeholder),
        4 : lambda : Generate_interview_questions(Assistant,placeholder)

    }
    step_function = step_functions.get(step)
    if step_function:
        step_function()
        

def display_conversation_history():
        st.write("Conversation History:")
        st.write(json.dumps(st.session_state.conversation_history, indent=2))
def main():
    if "cv_uploaded" not in st.session_state:
      st.session_state.cv_uploaded = False
    if "job_selected" not in st.session_state:
        st.session_state.job_selected = False
    if "CV_verified" not in st.session_state:
        st.session_state.CV_verified = False
    if "First_Interview_quesrion_answered"  not in st.session_state:
        st.session_state.First_Interview_quesrion_answered = False
    if "Second_interview_quesrion" not in st.session_state:
        st.session_state.Second_interview_quesrion = False
    if "third_interview_question" not in st.session_state:
        st.session_state.third_interview_question = False
    if "grade_conversation" not in st.session_state:
        st.session_state.grade_conversation = False
    if "interview_result" not in  st.session_state:
        st.session_state.interview_result = False 
    if "coding_test_done"  not in st.session_state:
        st.session_state.coding_test_done = False
    if "final_evaluation_done" not in  st.session_state:
        st.session_state.final_evaluation_done = False

    Assistant = InterviewerAssistant()
    company_name = "Arbisoft"
    st.title(f"{company_name} AI Interviewer")
    CodeEvaluation = Code_Evaluator()

    # Placeholder container for each step
    step_placeholder = st.empty()

    # Step 1: CV upload
    step_placeholder.text("Please upload your CV to start the interview process.")
    handle_interview_step(1, Assistant, step_placeholder)
    #step 2: Display all the available jobs
    if st.session_state.cv_uploaded:
        step_placeholder.empty()
        handle_interview_step(2, Assistant, step_placeholder)
    if st.session_state.job_selected:
        step_placeholder.empty()
        time.sleep(10)
        handle_interview_step(3, Assistant, step_placeholder)
    if st.session_state.CV_verified and not st.session_state.First_Interview_quesrion_answered:
        step_placeholder.empty()
        Generate_interview_questions(1,Assistant,step_placeholder,"Question_1")
    if st.session_state.First_Interview_quesrion_answered and  not st.session_state.Second_interview_quesrion:
        step_placeholder.empty()
        Generate_interview_questions(2,Assistant,step_placeholder,"Question_2")
    if st.session_state.Second_interview_quesrion and not st.session_state.third_interview_question:
        step_placeholder.empty()
        Generate_interview_questions(3,Assistant,step_placeholder,"Question_3")
    if st.session_state.third_interview_question and not  st.session_state.grade_conversation:
        Grader_bot = Grader()
        step_placeholder.empty()
        interview_data = Grader_bot.extract_interview_data(Assistant,st.session_state.conversation_history)
        print(interview_data)
        grade = Grader_bot.grade(Assistant,interview_data)
        print(grade)
        st.session_state.grade = grade
        Assistant.add_message_to_conversation("user",grade)
        marks = Assistant.get_any_data_fromtext(grade,"From this data extract the smallest number between 1 to 10 and return it as python integer/")
        print(f"Marks = {marks}")
        Assistant.add_message_to_conversation("user",f"marks obtained:{marks}")
    if st.session_state.grade_conversation and not  st.session_state.interview_result:
        step_placeholder.empty()
        st.text_area("Assistant",grade,height=100,disabled=True)
        try:
            if int(marks) > 5:
                st.success(f"You have a good score in the interview that is {marks}")
                time.sleep(10)
            else:
                st.error(f"You have a low score of {marks} in the interview try to do better in the coding test")
                time.sleep(10)
        except ValueError:
            st.write("")
        with st.spinner("Moving to the coding interview"):
            time.sleep(10)
        st.session_state.interview_result = True
    if st.session_state.interview_result and not st.session_state.coding_test_done:
        step_placeholder.empty()
        Generate_coding_questions(Assistant,step_placeholder)
    if st.session_state.coding_test_done and not st.session_state.final_evaluation_done:
        step_placeholder.empty()
        with st.spinner("Please wait for one minute so we can give you your final score"):
          time.sleep(10)
        print(st.session_state.Programming_questions_history)
        print(type(st.session_state.Programming_questions_history))
        code = CodeEvaluation.Evaluate_questions(st.session_state.Programming_questions_history)
        st.session_state.code = code
        print(code)
        st.text_area("CodingQuestionevaluation",code,height=100,disabled=True)
        st.session_state.final_evaluation_done = True
        with st.spinner("YOU will be sent an email with your final score in less than 5 mins"):
            time.sleep(20)
    if st.session_state.final_evaluation_done:
     Emailgenaration = Final_Email_generator()
     email = Emailgenaration.generate_email(st.session_state.grade, st.session_state.code)
     print(email) 
    
     my_email = #add your email
     your_password = #add your gmail app password
    
     try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as connection:
            connection.login(user=my_email, password=your_password)
            connection.sendmail(
                from_addr=my_email,
                to_addrs=st.session_state.user_email_adress,
                msg=email
            ) 
        st.text_area("Assistant","Email sent",height=100,disabled=True)
     except Exception as e:
        st.write("unable to send email here is the result")
        st.text_area("final result",email,height=100,disabled=True)

if __name__ == "__main__":
    main()