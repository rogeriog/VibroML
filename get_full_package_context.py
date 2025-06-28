import os

def concatenate_project_files(root_dir, output_file=None):
    """
    Concatenates the content of specified project files into a single string,
    prefixed with their relative paths.

    Args:
        root_dir (str): The root directory of your project.
        output_file (str, optional): If provided, the concatenated content
                                     will be written to this file.
                                     Otherwise, it will be printed to stdout.
    Returns:
        str: The concatenated content.
    """
    # Define the files and directories to include.
    # Order them logically for the LLM (e.g., README first, then setup, then source).
    files_to_include = [
        'README.md',
        # 'setup.py',
        # 'requirements.txt',
        'vibroml/main.py',
        'vibroml/auto_optimize.py',
        'vibroml/default_settings.json',
        'vibroml/utils/config.py',
        # 'vibroml/utils/__init__.py',
        'vibroml/utils/phonon_utils.py',
        'vibroml/utils/plotting_utils.py',
        'vibroml/utils/relaxation_utils.py',
        'vibroml/utils/structure_utils.py',
        'vibroml/utils/utils.py',
    ]

    concatenated_content = []

    for file_path in files_to_include:
        full_path = os.path.join(root_dir, file_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                concatenated_content.append(f"--- FILE: {file_path} ---\n")
                concatenated_content.append(content)
                concatenated_content.append("\n--- END FILE: {file_path} ---\n\n")
            except Exception as e:
                concatenated_content.append(f"--- ERROR READING FILE: {file_path} - {e} ---\n\n")
        else:
            concatenated_content.append(f"--- FILE NOT FOUND: {file_path} ---\n\n")

    final_string = "".join(concatenated_content)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_string)
        print(f"Concatenated content written to {output_file}")
    else:
        print(final_string)

    return final_string

if __name__ == "__main__":
    # Get the current working directory as the root
    project_root = os.getcwd()

    # Option 1: Print to console (default)
    # concatenate_project_files(project_root)

    # Option 2: Write to a file (recommended for LLM input)
    concatenate_project_files(project_root, output_file="vibroml_full_package_context.txt")

    print("\nScript finished. Check the console or 'vibroml_full_package_context.txt'.")