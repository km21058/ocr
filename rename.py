import os

def rename_files(folder_path):
    # Get the list of files in the folder
    file_list = os.listdir(folder_path)
    sorted_file_list = sorted(file_list)
    # Loop through each file and rename it
    """
    for filename in sorted_file_list:
        print(filename + "\n")
    """
    for index, filename in enumerate(sorted_file_list):
        # Create the new filename by removing numbers and underscores
        new_filename =  1 + index
        
        # Get the full path of the old and new filenames
        old_file_path = os.path.join(folder_path, filename)
        new_file_path = os.path.join(folder_path, str(new_filename) + ".pdf")
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
    
if __name__ == "__main__":
    folder_path = "./image/NewData"  # Replace with the actual path to your folder
    rename_files(folder_path)