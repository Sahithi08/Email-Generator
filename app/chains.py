import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama-3.1-70b-versatile"
    )
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE: 
            {page_data} 
            ### INSTRUCTION: 
            The scraped text is from the career's page of a website. 
            Also extract the info about that company and personalize mails using that info.
            Your job is to extract the job postings and return them in JSON format containing
            following keys: 'role', 'experience', 'skills' and 'description'. 
            Only return the valid JSON. 
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data': cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links, tone="formal"):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:  
            {job_description}  

            ### INSTRUCTION:  
            You are Sakshi, a business development executive at XYZ. XYZ is a service-based company known for delivering end-to-end solutions tailored to meet diverse client needs across industries. From efficient project management to scalable cloud solutions, XYZ specializes in helping organizations optimize operations, reduce costs, and achieve their goals seamlessly.
            Recently, XYZ has also begun incorporating AI-driven tools into its service offerings to enhance decision-making and streamline workflows further.
            Your job is to craft a cold email to the client regarding the job mentioned above, showcasing XYZ’s expertise and emphasizing how the company can address their specific requirements.
            The tone of the email should be **{tone}**.
            If "formal," write the email professionally with a structured approach.
            If "casual," write in a friendly, approachable manner with less rigid phrasing.
            Ensure each email feels unique and tailored to the client’s needs, avoiding repetitive phrasing or rigid templates.
            Also, include the most relevant examples from the following links to showcase XYZ’s portfolio: {link_list}  
            Remember, you are Sakshi, BDE at XYZ. Do not include a preamble or any additional closing remarks.  
            ### EMAIL (NO PREAMBLE):  
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links, "tone": tone})
        return res.content


if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))