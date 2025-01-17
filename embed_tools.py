import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import requests
import warnings
import os
from collections import defaultdict
import numpy as np
from qwen_vl_utils import process_vision_info
import json

Image.MAX_IMAGE_PIXELS = None   # disable DecompressionBombWarning
warnings.filterwarnings("ignore", category=FutureWarning)

processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1")
vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1", trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1')
text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
text_model.eval()

image_tuples = []

def load_json_data(file_path):
  """Loads JSON data from the specified file path into a global variable.

  Args:
    file_path: The path to the JSON file.
  """

  global image_tuples

  with open(file_path, 'r') as f:
    data = json.load(f)
    image_list = [(item['location'], item['embedding'], item['caption']) for item in data]

  return image_list

def normalize_image(image, max_width, max_height):
  """
  Resizes an oversized PIL image down to the maximum dimensions while maintaining aspect ratio.

  Args:
      image: A PIL Image object.
      max_width: Maximum allowed width for the resized image.
      max_height: Maximum allowed height for the resized image.

  Returns:
      A resized PIL Image object or the original image if not oversized.
  """
  width, height = image.size

  if width <= max_width and height <= max_height:
    return image

  if width / max_width > height / max_height:
    ratio = max_width / width
  else:
    ratio = max_height / height

  new_width = int(width * ratio)
  new_height = int(height * ratio)
  resized_image = image.resize((new_width, new_height), Image.LANCZOS)

  return resized_image

def progress_bar(current, total, bar_length=20):
  """
  Prints a simple progress bar to the terminal.

  Args:
    current: The current progress.
    total: The total progress.
    bar_length: The length of the progress bar.
  """

  percent = round((current / total) * 100)
  filled_length = int(bar_length * current // total)
  bar = '█' * filled_length + '-' * (bar_length - filled_length)
  print(f'\rProgress: |{bar}| {percent}%', end='')

def embed_images(image_sources, max_width=1024, max_height=1024):
    """
    Embeds a list of image sources (URLs or local file paths) in a batch and returns a list of tuples (source, embedding).
    Args:
        image_sources: A list of image sources (URLs or local file paths).
        max_width: Maximum width to resize images to
        max_height: Maximum height to resize images to
    Returns:
        A list of tuples (source, embedding), where embedding is a normalized tensor.
    """
    processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1")
    vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1", trust_remote_code=True)
    vision_model.eval()
    
    image_data = defaultdict(dict)
    progress_count = 0
    progress_total = len(image_sources)
    print("Processing images")
    
    # Process all images first
    for source in image_sources:
        try:
            if source.startswith("http"):
                response = requests.get(source, stream=True).raw
                image = Image.open(response)
            else:
                image = Image.open(source)
            image = normalize_image(image, max_width, max_height)
            progress_bar(progress_count, progress_total)
            image_data[source]["image"] = processor(image, return_tensors="pt")
        except Exception as e:
            print(f"\nError processing image from {source}: {e}")
            image_data[source]["error"] = True
        progress_count += 1

    progress_bar(progress_count, progress_total)
    # Check if we have any valid images
    valid_sources = [source for source, data in image_data.items() if not data.get("error")]
    if not valid_sources:
        return []

    print("\nEmbedding images")
    
    # Prepare the batch input
    pixel_values = torch.cat([
        image_data[source]["image"]["pixel_values"] 
        for source in valid_sources
    ], dim=0)
    
    # Process the batch
    with torch.no_grad():
        outputs = vision_model(pixel_values=pixel_values)
        img_embeddings = outputs.last_hidden_state
        # Normalize the embeddings
        img_embeddings = F.normalize(img_embeddings[:, 0], p=2, dim=1)

    # Create the final list of (source, embedding) tuples
    result = []
    for idx, source in enumerate(valid_sources):
        result.append((source, img_embeddings[idx]))
    
    return result

def generate_image_captions(image_paths, model_path="Qwen/Qwen2-VL-7B-Instruct", user_text="Describe the image in English.", device="cuda"):
    """Generates captions for a list of images using Qwen2-VL-7B.

    Args:
        image_paths (list): A list of image file paths or URLs.
        model_path (str, optional): The path to the model checkpoint. Defaults to "Qwen/Qwen2-VL-7B-Instruct".
        device (str, optional): The device to use for computation. Defaults to "cuda".

    Returns:
        list: A list of tuples (image_path, caption).
    """

    device = torch.device(device)

    # Load the model and tokenizer
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, _attn_implementation="flash_attention_2")

    # Move the model to the specified device (GPU)
    model.to(device)

    progress_count = 0
    progress_total = len(image_paths)
    print("Captioning images.")

    captions = []
    for image_path in image_paths:
        try:
            # Process the image
            progress_bar(progress_count, progress_total)
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Normalize size
            image = normalize_image(image, max_width=1024, max_height=1024)
            
            # Ensure we have 3 channels (RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Process with the model
            conversation = []
            user_content = []
            user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": user_text})
            conversation.append({"role": "user", "content": user_content})
            prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
            
            # Generate the caption
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=400)
            
            result = processor.decode(outputs[0], skip_special_tokens=True)
            caption = result.split('\nassistant\n', 1)[1]
            captions.append((image_path, caption))
            
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
            captions.append((image_path, f"Error: {str(e)}"))
        
        progress_count += 1

    progress_bar(progress_count, progress_total)
    print("\n")
    return captions

def get_images_in_directory(directory_path):
    """
    Returns a list of image file paths within a given directory.

    Args:
        directory_path: The path to the directory to search.

    Returns:
        A list of image file paths.
    """

    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more as needed
    image_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
              image_files.append(os.path.join(root, file))

    return image_files

def match_and_export_image_data(image_directory, output_file="image_data.json"):
    """
    Matches image embeddings and captions based on image paths, then exports data to JSON.

    Args:
        image_directory (str): Path to the directory containing images.
        output_file (str, optional): Path to the output JSON file. Defaults to "image_data.json".
    """

    # Get image paths
    image_paths = get_images_in_directory(image_directory)

    # Generate embeddings and captions
    embeddings = embed_images(image_paths)
    captions = generate_image_captions(image_paths)

    # Create a dictionary mapping image paths to their embeddings
    embedding_dict = {path: embedding.cpu().numpy().tolist() for path, embedding in embeddings}

    # Match embeddings and captions based on image paths
    matched_data = []
    for image_path, caption in captions:
        if image_path in embedding_dict:
            matched_data.append({"location": image_path, "embedding": embedding_dict[image_path], "caption": caption})

    # Export data to JSON
    with open(output_file, "w") as f:
        json.dump(matched_data, f, indent=4)

    print(f"Exported matched image data to {output_file}")


#embeds = embed_images(paths)
#match_and_export_image_data("/media/prev/home/cris/Pictures/chud")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def contrastive_search(query, tokenizer=tokenizer, text_model=text_model):
    """
    Performs contrastive search between a query and a list of image embeddings.

    Args:
        tokenizer: Tokenizer instance.
        text_model: Text model instance.
        query: The search query string.
        image_embeddings: A list of tuples, where the second element is an image embedding tensor.

    Returns:
        A list of tuples, where each tuple contains the image embedding and its similarity score to the query.
    """

    # Retrieve images
    image_list = load_json_data("/home/cris/Documents/text-generation-webui/extensions/reactive_image/image_data.json")

    # Tokenize and encode the query
    encoded_query = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        query_embedding = text_model(**encoded_query)

    # Mean pooling and normalization
    query_embedding = mean_pooling(query_embedding, encoded_query['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1)

    # Calculate similarity scores
    similarity_scores = []

    for img in image_list:
        #print(img[1])
        #print(torch.tensor(img[1]))
        similarity = torch.matmul(query_embedding, torch.tensor(img[1]).T)
        similarity_scores.append((img[0], img[2], similarity.item()))

    # Sort by similarity scores
    similarity_scores.sort(key=lambda x: x[2], reverse=True)
    #print(similarity_scores)
    return similarity_scores

image_tuples = load_json_data("/home/cris/Documents/text-generation-webui/extensions/reactive_image/image_data.json")
#print(image_tuples[1][2])
"""
#print(image_tuples)
#embeddings = [tuple[1] for tuple in image_tuples]

scores = contrastive_search(tokenizer, text_model, "search_query: Total tranny destruction", image_tuples)
for score in scores:
    print(score[0] + ": " + str(score[2]) + "\n" + score[1])
    print("\n\n")
"""

"""
sentences = ['search_query: Who was the leader of Germany during WWII?', 'search_query: What do cats look like?']

tokenizer = AutoTokenizer.from_pretrained('nomic-ai/nomic-embed-text-v1')
text_model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
text_model.eval()

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = text_model(**encoded_input)

text_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

print(torch.matmul(img_embeddings, text_embeddings.T))
"""

