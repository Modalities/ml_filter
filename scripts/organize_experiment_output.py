
import os
import shutil
import re

def organize_experiment_output(folder_path, educational_prompt="edu"):
    """
    Organize files by creating folders based on <model_name> and <lang_name>, and rename files
    according to the specified naming convention.

    Args:
        folder_path (str): Path to the folder containing the files.
        educational_prompt (str): The text to replace in the file name.
        en (str): The language code to include in the renamed files.
    """
    # Check if the provided exists and points to a directory
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder path {folder_path} does not exist.")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path {folder_path} is not a directory.")

    # Iterate through the files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".jsonl"):
            # Delete files starting with 'test_data_ml_filter_511__annotations_'
            if file_name.startswith("test_data_ml_filter_511__annotations_") and file_name.endswith(".jsonl"):
                file_to_delete = os.path.join(folder_path, file_name)
                try:
                    os.remove(file_to_delete)
                    print(f"Deleted file: {file_to_delete}")
                except Exception as e:
                    print(f"Error deleting {file_to_delete}: {e}")
            # Extract <lang_name> and <model_name> using regex
            match = re.match(r".*__annotations_(.*?)_fine_web_edu_(.*?)_.jsonl", file_name)
            if match:
                model_name = match.group(1)
                lang_name = match.group(2)
            else:
                raise NotImplementedError(f"File name format not recognized: {file_name}")
                # Special case for `en` folder
                match_en = re.match(r"511_test_documents_educational_content_en__annotations_(.*?)_fine_web_edu_en_.jsonl", file_name)
                if match_en:
                    lang_name = "en"
                    model_name = match_en.group(1)
                else:
                    continue  # Skip files that don't match

            # Create the new folder for the <model_name>
            model_folder_path = os.path.join(folder_path, model_name)
            os.makedirs(model_folder_path, exist_ok=True)

            # Create the new folder for the <lang_name> inside <model_name>
            lang_folder_path = os.path.join(model_folder_path, lang_name)
            os.makedirs(lang_folder_path, exist_ok=True)

            # Define the new file name
            new_file_name = f"annotations_{educational_prompt}_{lang_name}_{model_name}.jsonl"

            # Move and rename the file
            source_file_path = os.path.join(folder_path, file_name)
            destination_file_path = os.path.join(lang_folder_path, new_file_name)
            shutil.move(source_file_path, destination_file_path)

            print(f"Moved and renamed: {source_file_path} -> {destination_file_path}")
        
def find_generated_annotations(root_dir):
    generated_paths = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if 'generated_annotations' in dirnames:
            generated_paths.append(os.path.join(dirpath, 'generated_annotations'))
    return generated_paths


root_directory = "/raid/s3/opengptx/user/richard-rutmann/data/ml_filter/gemma-3-27b-it"
paths = find_generated_annotations(root_directory)        

for path in paths:
    organize_experiment_output(path)
    print(f"Organized files in: {path}")
