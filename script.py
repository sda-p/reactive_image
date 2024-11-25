import gradio as gr
import torch
from transformers import LogitsProcessor
import os
import re
from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply, generate_reply_HF, 
    generate_reply_custom,
)
from .embed_tools import generate_image_captions, load_json_data, contrastive_search
from collections import defaultdict
import base64
from PIL import Image
import io


params = {
    "display_name": "LLM Web Search",
    "is_tab": True,
    "enable": True,
    "search results per query": 5,
    "langchain similarity score threshold": 0.5,
    "instant answers": True,
    "regular search results": True,
    "search command regex": "",
    "default search command regex": r"Search_web\(\"(.*)\"\)",
    "open url command regex": "",
    "default open url command regex": r"Open_url\(\"(.*)\"\)",
    "display search results in chat": True,
    "display extracted URL content in chat": True,
    "searxng url": "",
    "cpu only": True,
    "chunk size": 500,
    "duckduckgo results per query": 10,
    "append current datetime": False,
    "default system prompt filename": None,
    "force search prefix": "Search_web",
    "ensemble weighting": 0.5,
    "keyword retriever": "bm25",
    "splade batch size": 2,
    "chunking method": "character-based",
    "chunker breakpoint_threshold_amount": 30
}


custom_system_message_filename = None
extension_path = os.path.dirname(os.path.abspath(__file__))
chat_id = None
force_search = False
update_history = defaultdict(str)
query_regex = r"\[(image\squery):\s(.+?)\]"
select_regex = r"\[(image\sselect):\s(.+?)\]"
thumbnail = None

def create_thumbnail(file_uri, max_size=(600, 600)):
    """
    Create a thumbnail from the given file URI and return the base64 image data 
    and an HTML <img> tag for rendering in a chat context.
    
    Args:
        file_uri (str): Path to the image file.
        max_size (tuple): Maximum dimensions (width, height) of the thumbnail.
        
    Returns:
        dict: A dictionary containing:
            - "image_data": Base64-encoded image data of the thumbnail.
            - "html_img": HTML <img> block for embedding the image.
    """
    try:
        # Open the image file
        image = Image.open(file_uri)
        
        # Create the thumbnail
        image.thumbnail(max_size)
        
        # Save the thumbnail to a buffer
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        buffered.seek(0)
        
        # Get the base64-encoded data
        image_bytes = buffered.getvalue()
        image_data = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create the <img> block
        html_img = f'<img src="data:image/jpeg;base64,{image_data}" alt="Thumbnail" style="max-width: {max_size[0]}px; max-height: {max_size[1]}px;">'
        
        return {
            "image_data": image_data,
            "html_img": html_img
        }
    except Exception as e:
        return {"error": str(e)}

def setup():
    """
    Is executed when the extension gets imported.
    :return:
    """
    global params
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["QDRANT__TELEMETRY_DISABLED"] = "true"

    #try:
        #with open(os.path.join(extension_path, "settings.json"), "r") as f:
            #saved_params = json.load(f)
        #params.update(saved_params)
        #save_settings()   # add keys of newly added feature to settings.json
    #except FileNotFoundError:
        #save_settings()

    #if not os.path.exists(os.path.join(extension_path, "system_prompts")):
        #os.makedirs(os.path.join(extension_path, "system_prompts"))

    #toggle_extension(params["enable"])

def history_modifier(history):
    """
    Modifies the chat history.
    Only used in chat mode.
    """
    return history

def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """
    return state

def chat_input_modifier(text, visible_text, state):
    """
    Modifies the user input string in chat mode (visible_text).
    You can also modify the internal representation of the user
    input (text) to change how it will appear in the prompt.
    """
    return text, visible_text

def input_modifier(string, state, is_chat=False):
    """
    In default/notebook modes, modifies the whole prompt.

    In chat mode, it is the same as chat_input_modifier but only applied
    to "text", here called "string", and not to "visible_text".
    """
    global thumbnail
    thumbnail = None
    return string

def bot_prefix_modifier(string, state):
    """
    Modifies the prefix for the next bot reply in chat mode.
    By default, the prefix will be something like "Bot Name:".
    """
    return string

def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    """
    Modifies the input ids and embeds.
    Used by the multimodal extension to put image embeddings in the prompt.
    Only used by loaders that use the transformers library for sampling.
    """
    return prompt, input_ids, input_embeds

def logits_processor_modifier(processor_list, input_ids):
    """
    Adds logits processors to the list, allowing you to access and modify
    the next token probabilities.
    Only used by loaders that use the transformers library for sampling.
    """
    #processor_list.append(MyLogits())
    return processor_list

def output_modifier(string, state, is_chat=False):
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """
    modified_string = re.sub(r"(%=%(?:.*?\s*?)*?%=%)", "", string)
    modified_string = re.sub(r"(\[image\s.*?\])", "", modified_string)
    if thumbnail:
        modified_string = thumbnail["html_img"] + modified_string
    return modified_string


def custom_generate_reply(question, original_question, seed, state, stopping_strings, is_chat, recursive_call=False):
    """
    Custom generation function with tool invocation support and context updating.
    """
    global update_history  # If your framework uses global state for tracking chat history
    global thumbnail

    # Model-specific generation function
    generate_func = generate_reply_HF if shared.model.__class__.__name__ not in [
        'LlamaCppModel', 'RWKVModel', 'ExllamaModel', 'Exllamav2Model', 'CtransformersModel'] else generate_reply_custom

    # Regex for tool invocation
    query_command_regex = re.compile(query_regex)  # Example: [tool:argument]
    select_command_regex = re.compile(select_regex)
    local_dir = os.path.dirname(os.path.realpath(__file__))

    for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
        #print(f"Generated reply: {reply}")  # Debugging output

        # Check for tool invocation
        query_match = query_command_regex.search(reply)
        select_match = select_command_regex.search(reply)
        if query_match:
            tool_name, tool_argument = query_match.groups()
            yield reply  # Output the current reply containing the tool command
            original_model_reply = reply
            # Execute the tool
            print(f"Executing {tool_name} with argument: {tool_argument}")
            try:
                candidates = contrastive_search(tool_argument)
                top_k_candidates = candidates[:3] #!Parametrize this
                #print(candidates)
                result = "%=%\nOptions:\n"
                for candidate in top_k_candidates:
                    result += str(os.path.basename(candidate[0])) + ":\n" + candidate[1] + "\n"
                result += "%=%\n\n"
            except Exception as exc:
                result = f"%=%\nError executing {tool_name}: {str(exc)}\n%=%"
                print(f"Tool execution error: {exc}")

            # Append tool output to the context
            reply += f"\n{result}\n"

            # Update the context/question and recursively call for continuation
            new_question = f"{question}\n{reply}"  # Or update `state['history']` if applicable
            new_reply = ""
            for new_reply in custom_generate_reply(new_question, new_question, seed, state, stopping_strings,
                                                   is_chat=is_chat, recursive_call=True):
                yield f"{reply}\n{new_reply}"
            return  # Exit after recursive call completes to avoid redundant processing

        if select_match:
            tool_name, tool_argument = select_match.groups()
            yield reply  # Output the current reply containing the tool command
            original_model_reply = reply
            # Execute the tool
            print(f"Executing {tool_name} with argument: {tool_argument}")

            file_location = local_dir + "/library/" + tool_argument
            print(file_location)

            if os.path.isfile(file_location):
                thumbnail = create_thumbnail(file_location)
                result = f'%=%\nImage attached successfully.\n%=%\n'
            else:
                result = f"%=%\nError executing {tool_name}: malformed file name.\n%=%"

            reply += f"\n{result}\n"
            # Update the context/question and recursively call for continuation
            new_question = f"{question}\n{reply}"  # Or update `state['history']` if applicable
            new_reply = ""
            for new_reply in custom_generate_reply(new_question, new_question, seed, state, stopping_strings,
                                                   is_chat=is_chat, recursive_call=True):
                yield f"{reply}\n{new_reply}"
            return  # Exit after recursive call completes to avoid redundant processing

        yield reply  # Yield reply if no tool invocation detected

    # Optional: Update history for recursive calls
    if recursive_call:
        update_history[state["unique_id"]] = f"{reply}\n{update_history.get(state['unique_id'], '')}"

def custom_css():
    """
    Returns a CSS string that gets appended to the CSS for the webui.
    """
    return ''

def custom_js():
    """
    Returns a javascript string that gets appended to the javascript
    for the webui.
    """
    return ''

def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.

    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """
    pass
