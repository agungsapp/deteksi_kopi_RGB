import os

def rename_files_in_directory(directory_path):
    """Renames all files in a directory with a sequential number in parentheses based on the directory name.

    Args:
        directory_path: The path to the directory containing the files.
    """

    directory_name = os.path.basename(directory_path)
    file_count = 1
    for filename in os.listdir(directory_path):
        old_path = os.path.join(directory_path, filename)
        new_filename = f"{directory_name}({file_count}).png"
        new_path = os.path.join(directory_path, new_filename)
        os.rename(old_path, new_path)
        file_count += 1


directories = [
    "dataset/test/A",
    "dataset/test/B",
    "dataset/test/C",
    "dataset/train/A",
    "dataset/train/B",
    "dataset/train/C",
   
]


for directory in directories:
    rename_files_in_directory(directory)

print("Renaming completed successfully.")