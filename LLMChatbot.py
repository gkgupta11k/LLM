#LLm

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def chat_with_gpt2():
    # Load pre-trained model and tokenizer with padding_side set to left
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium", padding_side='left')
    model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
    model.eval()  # Set to evaluation mode to speed up

    print("Chat with GPT-2! Type 'exit' to end.")
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        # Encode and add special tokens
        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

        # Generate response
        with torch.no_grad():
            response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
        
        # Decode and print the response
        response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        print(f"GPT-2: {response}")

if __name__ == "__main__":
    chat_with_gpt2()
