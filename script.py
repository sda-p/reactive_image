import gradio as gr
import torch
from transformers import LogitsProcessor
import os
import re
from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)
from embed_tools import generate_image_captions, load_json_data


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

ddef history_modifier(history):
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
    processor_list.append(MyLogits())
    return processor_list

def output_modifier(string, state, is_chat=False):
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """
    return string

def custom_generate_reply(question, original_question, seed, state, stopping_strings, is_chat, recursive_call=False):
    """
    Minimal implementation for handling arbitrary tool calls and updating history.
    """
    global update_history
    
    # Placeholder for a text generation function
    generate_func = generate_reply  # Replace with appropriate model-specific function

    # Compile regex for tool invocation
    tool_command_regex = re.compile(r"\[(tool):(.+?)\]")  # Example pattern: [tool:argument]
    
    reply = None
    for reply in generate_func(question, original_question, seed, state, stopping_strings, is_chat=is_chat):
        
        # Check for tool invocation
        tool_match = tool_command_regex.search(reply)
        if tool_match:
            tool_name, tool_argument = tool_match.groups()
            yield reply  # Output the current reply containing the tool invocation

            # Dummy tool execution placeholder
            print(f"Executing {tool_name} with argument: {tool_argument}")
            tool_output = f"Output from {tool_name} with argument '{tool_argument}'"

            # Modify reply with tool output (placeholder)
            reply += f"\n[Tool Result]: {tool_output}"

            # Optional continuation with modified context
            new_question = f"{question} {reply}"  # Update the context for recursion or follow-up
            new_reply = ""
            for new_reply in custom_generate_reply(new_question, new_question, seed, state,
                                                   stopping_strings, is_chat=is_chat, recursive_call=True):
                yield new_reply
            break
        print("Invoked.")
        
        yield reply  # Regular output if no tool invocation detected

    # Update history if in a recursive call
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
