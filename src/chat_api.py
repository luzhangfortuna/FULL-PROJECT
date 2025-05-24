
import time
import datetime
import pandas as pd
from openai import OpenAI
from .models import db, ChatMessage
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv('keys.env')

# Get the OpenAI key from environment variables
openai_key = os.getenv('OPENAI_API_KEY')

if openai_key:
    print("OpenAI key loaded successfully!")
else:
    print("OpenAI key not found.")

# Initialize OpenAI client
client = OpenAI()

# Path to your CSV file
csv_file_path = "/Users/zhanglu/Desktop/acquisitions_update_2021.csv"

# Construct the system prompt
prompt_template = """ Today is {today}.  
You are Luna, a virtual market researcher specializing in the tech industry, created by LU.  
Your expertise lies in analyzing market trends, mergers and acquisitions (M&A) activities, and providing insights into strategic opportunities within the tech sector.  

You deliver responses that are professional, concise, and backed by relevant data. Your tone reflects a deep understanding of industry dynamics, emphasizing clarity and practicality. Answer each question truthfully to the best of your abilities based on the provided information, focusing on:  

- Market trends and growth opportunities in the tech sector.  
- M&A activities, including key players, motivations, and outcomes.  
- Strategic insights derived from market data and case studies.  
- Competitive landscape analysis within the tech industry.  
- Potential risks and opportunities linked to M&A transactions.  

Structure your responses as follows:  
1. Brief summary of the key point(s).  
2. Detailed insights presented in bullet points, highlighting actionable recommendations or implications where applicable.  
3. Where relevant, include examples or data points to contextualize your analysis.  

Your goal is to support businesses, investors, and stakeholders in making informed decisions in the tech industry's evolving market.  

<context>  
{context}  
</context>  
"""



# Function to fetch data from CSV
def get_csv_data(file_path):
    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Ensure that the DataFrame contains the columns you need (e.g., 'Acquired Company', 'Acquisition Price')
        required_columns = ['ID', 'Parent Company', 'Acquisition Year', 'Acquisition Month', 'Acquired Company', 'Acquisition Price']
        
        # Check if required columns exist
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain the following columns: {', '.join(required_columns)}")

        # Combine relevant columns into a formatted text
        text_list = [
            f"Acquisition: {row['Acquired Company']} ({row['Acquisition Year']}-{row['Acquisition Month']})\n"
            f"Parent Company: {row['Parent Company']}\n"
            f"Acquisition Price: {row['Acquisition Price']}\n"
            for _, row in df.iterrows()
        ]
        
        # Combine all text entries into a single string
        return "\n\n".join(text_list)
    
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return ""

import pandas as pd

# Fetch context from the CSV file
full_text = get_csv_data(csv_file_path)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Chat function
def call_chat(question):
    try:
        # Create the system prompt with context and current timestamp
        system_prompt = prompt_template.format(today=now, context=full_text)

        # Call OpenAI's chat completion API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=500,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            stream=True,
        )

        # Buffer to store the response
        answer_buffer = ""

        # Stream response and yield tokens
        for event in response:
            token = event.choices[0].delta.content
            if token:
                answer_buffer += token
                yield {"token": token}

        # Save the chat message to the database
        chat_message = ChatMessage(user_id=1, question=question, answer=answer_buffer)
        db.session.add(chat_message)
        db.session.commit()
    except Exception as e:
        print(f"Error in call_chat function: {e}")
