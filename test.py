from utils import pdf_to_images , test_preprocessing , line_splitter , extract_bboxes
import os
import torch
from transformers import GenerationConfig
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model
import cv2
from tqdm import tqdm
import natsort

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
from transformers import TrOCRProcessor, AutoTokenizer, VisionEncoderDecoderModel
processor = TrOCRProcessor.from_pretrained("qantev/trocr-base-spanish" , use_fast = True)
model = VisionEncoderDecoderModel.from_pretrained("qantev/trocr-base-spanish")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.decoder.config.vocab_size = processor.tokenizer.vocab_size
model.config.vocab_size = model.decoder.config.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id


config = LoraConfig(
    r=16,  # Rank (adjust as needed) #16
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout for LoRA layers
    target_modules=[
        "query" , "value" , "q_proj" , "v_proj"
    ]
)
# Apply LoRA
model = get_peft_model(model, config)

model.to(device)
checkpoint_path = "model_checkpoint.pth"  # Path to saved weights
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

gen_config = GenerationConfig(
    max_length=64,
    early_stopping=True,
    no_repeat_ngram_size=3,
    length_penalty=2.0,
    num_beams=4
)


           
def process_gen(line_dir, batch_size=16):
    image_files =  natsort.natsorted(os.listdir(line_dir))  # List all image files
    total_images = len(image_files)
    
    generated_texts = []  # Store generated texts
    
    for i in tqdm(range(0, total_images, batch_size), desc="Processing Batches"):
        batch_files = image_files[i : i + batch_size]  # Get batch of images
        
        # Load images and preprocess
        batch_images = [processor(cv2.imread(os.path.join(line_dir, img)), return_tensors="pt").pixel_values.squeeze() for img in batch_files]
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            outputs = model.generate(batch_tensor, generation_config=gen_config)
            batch_texts = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_texts.extend(batch_texts)  # Append results
    
        torch.cuda.empty_cache()  # Free up memory after each batch

    return generated_texts

import os

def save_text_only(generated_texts , output_txt_path):
    with open(output_txt_path, 'w') as f:
        for text in generated_texts:
            f.write(text)
            f.write('\n')
    print(f"✅ Text saved : {output_txt_path}")


pdf_dir = 'test_pdf'
indices = {filename: None for filename in os.listdir(pdf_dir) if filename.endswith('.pdf')}

for i in range(len(os.listdir(pdf_dir))):
    pdf = os.listdir(pdf_dir)[i]
    l = pdf_to_images(pdf_path = os.path.join(pdf_dir , pdf) , page_indices = indices[pdf] )
    for img_path in l:
        test_preprocessing(img_path) 
        line_splitter(img_path)
        json_path = f'results/surya/{(os.path.basename(img_path)).split(".")[0]}/results.json'
        sorted_bboxes = extract_bboxes(image_path = img_path , json_path = json_path , output_dir= os.path.splitext(os.path.basename(img_path))[0])
        if sorted_bboxes == None:
            continue
        gen_text = process_gen(os.path.splitext(os.path.basename(img_path))[0])
        os.makedirs('pred_transcriptions',exist_ok = True)
        save_text_only(gen_text , output_txt_path=os.path.join('pred_transcriptions',os.path.splitext(os.path.basename(img_path))[0]+'.txt'))

