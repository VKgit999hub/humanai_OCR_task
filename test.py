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
checkpoint_path = "/teamspace/studios/this_studio/2nd_qantev_model/model_checkpoint_epoch_9.pth"  # Path to saved weights
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

gen_config = GenerationConfig(
    max_length=64,
    early_stopping=True,
    no_repeat_ngram_size=3,
    length_penalty=2.0,
    num_beams=4
)


# images = [processor(cv2.imread('/teamspace/studios/this_studio/Buendia - Instruccion.pdf_page_1/cropped_11.png'), return_tensors="pt").pixel_values.squeeze() ]
# tensor = torch.stack(images).to(device)

# with torch.no_grad():
#     outputs = model.generate(tensor, generation_config=gen_config)
#     batch_texts = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     print(batch_texts)



           
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

def save_text_with_layout(generated_texts, bbox_positions, output_txt_path):
    """
    Saves the generated text in a text file while preserving the layout 
    based on bounding box coordinates.

    Parameters:
    - generated_texts: List of strings, each corresponding to a detected text line.
    - bbox_positions: List of tuples (x, y, w, h) representing bounding boxes.
    - output_txt_path: Path to save the generated text file.
    """

    with open(output_txt_path, 'w') as f:
        prev_y = None
        prev_x_end = 0

        for b, text in zip(bbox_positions, generated_texts):
            x1 , y1 , x2 , y2 = b['bbox']
            x = (x1+x2)/2
            y = (y1+y2)/2
            w = x2 - x1
            h = y2 - y1
            if prev_y is not None:
                # Insert a newline if the vertical gap is significant (new paragraph)
                if y - prev_y > 10:
                    f.write("\n")

                # Insert spaces proportional to the horizontal gap
                space_count = max(int(x - prev_x_end) // 10, 1)  # Adjust space scaling factor
                f.write(" " * space_count)

            f.write(text)
            prev_y = y
            prev_x_end = x + w

        f.write("\n")

    print(f"✅ Text saved with layout preservation: {output_txt_path}")



# 

pdf_dir = 'test_pdf'
indices = { 'Buendia - Instruccion.pdf' : None , 'Constituciones sinodales Calahorra 1602.pdf' : None ,'Ezcaray - Vozes.pdf' :None , 'Mendo - Principe perfecto.pdf' : None ,'Paredes - Reglas generales.pdf' : None,'PORCONES.228.35 – 1636.pdf' :None , "J&#x3a;0017&#x3a;03-J&#x3a;0085&#x3a;11 – 1799-1845.pdf" : None , "ES-AHPHU - J-000312-0014 – 1579.pdf": None}

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
        # print(gen_text)
        os.makedirs('pred_transcriptions',exist_ok = True)
        save_text_only(gen_text , output_txt_path=os.path.join('pred_transcriptions',os.path.splitext(os.path.basename(img_path))[0]+'.txt'))
        #save_text_with_layout(gen_text , sorted_bboxes ,output_txt_path=os.path.splitext(os.path.basename(img_path))[0]+'.txt'  )

