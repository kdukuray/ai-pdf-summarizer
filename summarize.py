import sys
from pypdf import PdfReader
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI



if (len(sys.argv) == 3):
	pdf_file_path = sys.argv[1]
	api_key_environment_variable = sys.argv[2]
else:
	raise Exception("Please pass in two command line arguments to the script. The first is your pdf_file_path, and the second is your api key environment varaible name")
	 
pdf = PdfReader(pdf_file_path)
pdf_text: str = ""
system_prompt = """
You are a text summarization AI creating a JSON summary:

Objectives:
- Capture text's core essence
- Prioritize critical information
- Remove redundant details

Output:
{
 "main_topic": "...",
 "key_points": [
   {
     "key_point": "...",
     "supporting_details": ["..."]
   }
 ],
 "additional_insights": ["..."]
}

"""
for page in pdf.pages:
	page_text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False) 
	pdf_text += page_text
	
splitter =  RecursiveCharacterTextSplitter(
	chunk_size=4000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
    )
    
split_text = splitter.split_text(pdf_text)

client = OpenAI(api_key=os.environ.get(api_key_environment_variable))
for chunk in split_text:
	response = client.chat.completions.create(
		model="gpt-4o-mini",
		messages=[
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": chunk}
		]
		)
		
	chunk_summary = response.choices[0].message.content
	try:
		chunk_as_dictionary = json.loads(chunk_summary)
		print("------------------------------------")
		print(f'Main Topic: {chunk_as_dictionary.get("main_topic")}')
		print()
		for key_point in chunk_as_dictionary.get("key_points"):
			print(f'{key_point.get("key_point")}')
			for supporting_details in key_point.get("supporting_details"):
				print(f"\t{supporting_details}")
			print()
		print("------------------------------------")
		print("------------------------------------")
	except Exception as e:
		print("An Error occured: ", e)