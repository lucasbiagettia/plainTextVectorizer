import os

def read_files_in_folder(folder):
    content_dictionary = {}

    try:
        for file_name in os.listdir(folder):
            if file_name.endswith(".txt"):
                full_path = os.path.join(folder, file_name)
                
                try:
                    with open(full_path, 'r') as file:
                        content = file.read()
                        content_dictionary[file_name] = content
                except Exception as e:
                    print(f"Error reading file {file_name}: {e}")

    except Exception as e:
        print(f"Error listing files in {folder}: {e}")

    return content_dictionary


