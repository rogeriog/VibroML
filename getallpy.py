import os

def collect_py_files(root_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as outfile:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(subdir, file)
                    outfile.write(f"\n\n# ==== File: {file_path} ====\n\n")
                    with open(file_path, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())

if __name__ == "__main__":
    root_directory = "."  # change to your target folder
    output_filename = "all_python_files.txt"
    collect_py_files(root_directory, output_filename)
    print(f"All .py files merged into {output_filename}")
