{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'transformers'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[2], line 4\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[39m#LLm\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[39mimport\u001b[39;00m \u001b[39mtorch\u001b[39;00m\n\u001b[0;32m----> 4\u001b[0m \u001b[39mfrom\u001b[39;00m \u001b[39mtransformers\u001b[39;00m \u001b[39mimport\u001b[39;00m GPT2LMHeadModel, GPT2Tokenizer\n\u001b[1;32m      6\u001b[0m \u001b[39mdef\u001b[39;00m \u001b[39mchat_with_gpt2\u001b[39m():\n\u001b[1;32m      7\u001b[0m     \u001b[39m# Load pre-trained model and tokenizer with padding_side set to left\u001b[39;00m\n\u001b[1;32m      8\u001b[0m     tokenizer \u001b[39m=\u001b[39m GPT2Tokenizer\u001b[39m.\u001b[39mfrom_pretrained(\u001b[39m\"\u001b[39m\u001b[39mgpt2-medium\u001b[39m\u001b[39m\"\u001b[39m, padding_side\u001b[39m=\u001b[39m\u001b[39m'\u001b[39m\u001b[39mleft\u001b[39m\u001b[39m'\u001b[39m)\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'transformers'"
     ]
    }
   ],
   "source": [
    "#LLm\n",
    "\n",
    "import torch\n",
    "from transformers import GPT2LMHeadModel, GPT2Tokenizer\n",
    "\n",
    "def chat_with_gpt2():\n",
    "    # Load pre-trained model and tokenizer with padding_side set to left\n",
    "    tokenizer = GPT2Tokenizer.from_pretrained(\"gpt2-medium\", padding_side='left')\n",
    "    model = GPT2LMHeadModel.from_pretrained(\"gpt2-medium\")\n",
    "    model.eval()  # Set to evaluation mode to speed up\n",
    "\n",
    "    print(\"Chat with GPT-2! Type 'exit' to end.\")\n",
    "    while True:\n",
    "        # Get user input\n",
    "        user_input = input(\"You: \")\n",
    "        if user_input.lower() == 'exit':\n",
    "            break\n",
    "\n",
    "        # Encode and add special tokens\n",
    "        input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors=\"pt\")\n",
    "\n",
    "        # Generate response\n",
    "        with torch.no_grad():\n",
    "            response_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)\n",
    "        \n",
    "        # Decode and print the response\n",
    "        response = tokenizer.decode(response_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)\n",
    "        print(f\"GPT-2: {response}\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    chat_with_gpt2()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.1"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}