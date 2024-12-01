from pathlib import Path
import yaml

from constants import EUROPEAN_LANGUAGES, TARGET_LANGAUGE_PLACEHOLDER
import os

def add_target_langauge_to_prompt(input_file_path: Path, output_dir: Path) -> None:
    """
    Reads a YAML file, replaces '{##TARGET_LANGUAGE##}' in the 'prompt' key with a given value, and writes the result to a new file.

    :param file_path: Path to the input YAML file.
    :param output_dir: Dir to save the updated YAML file.
    :param replacement: The string to replace '{##TARGET_LANGUAGE##}' with. Default is 'X'.
    """
    for language_code, language in EUROPEAN_LANGUAGES.items():
        try:
            # Read the YAML file
            with open(input_file_path, 'r') as file:
                data = yaml.safe_load(file)
        
            # Check if 'prompt' key exists
            if 'prompt' in data:
                original_prompt = data['prompt']
                updated_prompt = original_prompt.replace(TARGET_LANGAUGE_PLACEHOLDER , language)
                data['prompt'] = updated_prompt
            
                # Save the updated YAML
                file_name = input_file_path.stem
                output_file_path = output_dir /f"{file_name }_{language_code}.yaml"
                with open(output_file_path, 'w') as file:
                    yaml.safe_dump(data, file, default_flow_style=False)
            
                print(f"Updated 'prompt' saved to {output_dir}")
            else:
                 print("Key 'prompt' not found in the YAML file.") 
        except Exception as e:
            print(f"Teh following error occurred: {e}.")
   