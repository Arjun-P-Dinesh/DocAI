# import json
# import re
# from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# # Import the function from the other Python file
# from pdf_extractor import extract_text

# # Initialize the model and tokenizer
# model_name = 'EleutherAI/gpt-neo-125M'
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPTNeoForCausalLM.from_pretrained(model_name)

# def gpt_neo_autofill(field, user_data):
#     prompt = f"Given the following user data: {json.dumps(user_data)}, what should be the value for {field}?"
#     inputs = tokenizer(prompt, return_tensors='pt')
#     outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     value = generated_text.split(f"value for {field}?")[1].strip().split('\n')[0]
#     return value

# def autofill_text(extracted_text, user_data):
#     placeholders = re.findall(r'\{(.*?)\}', extracted_text)
#     for field in placeholders:
#         autofill_value = gpt_neo_autofill(field, user_data)
#         extracted_text = extracted_text.replace(f'{{{field}}}', autofill_value)
#     return extracted_text

# # Load user data from a JSON file
# with open('/home/arjgorthmic/SpecializationProject/samplePDF/userdata.json', 'r') as f:
#     user_data = json.load(f)

# # # Assume we are using a PDF named 'user_document.pdf'
# extracted_text = extract_text("/home/arjgorthmic/SpecializationProject/sample_document-1.pdf")

# # Use the autofill function to process the text
# autofilled_text = autofill_text(extracted_text, user_data)

# # Print the autofilled text
# print(autofilled_text)




import json
import re
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Import the function from the other Python file
from pdf_extractor import extract_text

# Initialize the model and tokenizer
model_name = 'EleutherAI/gpt-neo-125M'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

def gpt_neo_autofill(field, user_data):
    # Adjusted the prompt to clarify the request
    prompt = f"What should be the value for {field} given the user data: {json.dumps(user_data, indent=2)}"
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Improved logic to extract the relevant autofill value
    value = re.search(f"{field}:(.+)", generated_text)
    if value:
        return value.group(1).strip()
    else:
        return "No data found"  # Fallback in case no matching data found

def autofill_text(extracted_text, user_data):
    # Regex to find placeholders and their field names correctly
    placeholders = re.findall(r'([A-Za-z ]+): _+', extracted_text)
    for field in placeholders:
        field_clean = field.strip()
        autofill_value = gpt_neo_autofill(field_clean, user_data)
        # Updated regex to match the field text and underscores correctly
        extracted_text = re.sub(f"{field}: _+", f"{field}: {autofill_value}", extracted_text)
    return extracted_text

# Load user data from a JSON file
with open('/home/arjgorthmic/SpecializationProject/samplePDF/userdata.json', 'r') as f:
    user_data = json.load(f)

# Assume we are using a PDF named 'user_document.pdf'
extracted_text = extract_text("/home/arjgorthmic/SpecializationProject/sample_document-1.pdf")

# Use the autofill function to process the text
autofilled_text = autofill_text(extracted_text, user_data)

# Print the autofilled text
print(autofilled_text)
