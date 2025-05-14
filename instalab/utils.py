"""This module provides utility functions for various tasks.

- Compiling LaTeX code to PDF.
- Managing files and directories (removing figures, removing directories, saving files).
- Extracting specific content from text using regular expressions.

The module relies on standard Python libraries such as `os`, `re`, `shutil`,
and `subprocess` to perform these operations.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess


def compile_latex(
    latex_code: str, output_path: str, compile_tex: bool = True, timeout: int = 30
) -> str:
    """Compiles a given LaTeX string into a PDF document.

    The function first modifies the LaTeX code to include a predefined set of
    common LaTeX packages. It then writes this modified code to a .tex file
    and attempts to compile it using `pdflatex`.

    Args:
        latex_code (str): The LaTeX code string to be compiled.
        output_path (str): The base directory where the compilation artifacts
                           (including a 'tex' subdirectory for the .tex file
                           and output PDF) will be stored.
        compile_tex (bool, optional): If True, the LaTeX code will be compiled.
                                   If False, only the .tex file will be created,
                                   and compilation will be skipped. Defaults to True.
        timeout (int, optional): The maximum time in seconds to allow for the
                                 `pdflatex` compilation process. Defaults to 30.

    Returns:
        str: A message indicating the result of the compilation.
             This can be a success message with output, a timeout error,
             or a general compilation error message.
    """
    latex_code = latex_code.replace(
        r"\documentclass{article}",
        "\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{array}\n\\usepackage{algorithm}\n\\usepackage{algorithmicx}\n\\usepackage{algpseudocode}\n\\usepackage{booktabs}\n\\usepackage{colortbl}\n\\usepackage{color}\n\\usepackage{enumitem}\n\\usepackage{fontawesome5}\n\\usepackage{float}\n\\usepackage{graphicx}\n\\usepackage{hyperref}\n\\usepackage{listings}\n\\usepackage{makecell}\n\\usepackage{multicol}\n\\usepackage{multirow}\n\\usepackage{pgffor}\n\\usepackage{pifont}\n\\usepackage{soul}\n\\usepackage{sidecap}\n\\usepackage{subcaption}\n\\usepackage{titletoc}\n\\usepackage[symbol]{footmisc}\n\\usepackage{url}\n\\usepackage{wrapfig}\n\\usepackage{xcolor}\n\\usepackage{xspace}",
    )
    # print(latex_code)
    dir_path = f"{output_path}/tex"
    os.makedirs(dir_path, exist_ok=True)  # Ensure the directory exists
    tex_file_path = os.path.join(dir_path, "temp.tex")
    # Write the LaTeX code to the .tex file in the specified directory
    with open(tex_file_path, "w", encoding="utf-8") as f:
        f.write(latex_code)

    if not compile_tex:
        return "Compilation successful (file saved, compilation skipped)"

    # Compiling the LaTeX code using pdflatex with non-interactive mode and timeout
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", dir_path, "temp.tex"],
            check=True,  # Raises a CalledProcessError on non-zero exit codes
            capture_output=True,  # Replaces stdout=subprocess.PIPE and stderr=subprocess.PIPE
            timeout=timeout,  # Timeout for the process
            cwd=dir_path,  # Set working directory to where temp.tex is
        )

        # If compilation is successful, return the success message
        return f"Compilation successful: {result.stdout.decode('utf-8', errors='ignore')}"

    except subprocess.TimeoutExpired:
        # If the compilation takes too long, return a timeout message
        return f"[CODE EXECUTION ERROR]: Compilation timed out after {timeout} seconds"
    except subprocess.CalledProcessError as e:
        # If there is an error during LaTeX compilation, return the error message
        error_output = e.stderr.decode("utf-8", errors="ignore")
        stdout_output = e.stdout.decode(
            "utf-8", errors="ignore"
        )  # Also capture stdout for more context
        # Try to find the specific LaTeX error if possible
        log_file_path = os.path.join(dir_path, "temp.log")
        latex_error = ""
        if os.path.exists(log_file_path):
            with open(log_file_path, encoding="utf-8", errors="ignore") as log_file:
                log_content = log_file.read()
                error_match = re.search(
                    r"^!.*?\n(.*?)\n\nl\.\d+", log_content, re.MULTILINE | re.DOTALL
                )
                if error_match:
                    latex_error = error_match.group(0).strip()  # Get the error block
                else:  # Fallback if specific error pattern not found
                    first_error_line = re.search(r"^!.*", log_content, re.MULTILINE)
                    if first_error_line:
                        latex_error = first_error_line.group(0)

        if latex_error:
            return f"[CODE EXECUTION ERROR]: Compilation failed. LaTeX error:\n{latex_error}\n\nFull stdout:\n{stdout_output}\nFull stderr:\n{error_output}"
        return f"[CODE EXECUTION ERROR]: Compilation failed. There was an error in your latex. stdout:\n{stdout_output}\nstderr:\n{error_output}"


def remove_figures() -> None:
    """Removes PNG image files from the current working directory.

    This function iterates through all files in the current directory ('.')
    and deletes any file whose name starts with "Figure_" and ends with ".png".
    Prints a message for each removed file or if no such files are found.
    """
    removed_count = 0
    for _file in os.listdir("."):
        if "Figure_" in _file and ".png" in _file:
            try:
                os.remove(_file)
                print(f"Removed figure: {_file}")
                removed_count += 1
            except OSError as e:
                print(f"Error removing file {_file}: {e}")
    if removed_count == 0:
        print("No figures matching 'Figure_*.png' found to remove.")


def remove_directory(dir_path: str) -> None:
    """Removes a specified directory and all its contents.

    If the directory exists and is indeed a directory, it will be removed.
    Prints a message indicating success or failure, or if the directory
    does not exist.

    Args:
        dir_path (str): The path to the directory to be removed.
    """
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        try:
            shutil.rmtree(dir_path)
            print(f"Directory {dir_path} removed successfully.")
        except Exception as e:
            print(f"Error removing directory {dir_path}: {e}")
    else:
        print(f"Directory {dir_path} does not exist or is not a directory.")


def save_to_file(location: str, filename: str, data: str) -> None:
    """Saves a string of data to a plain text file at a specified location.

    The function constructs the full file path from the location and filename,
    then writes the provided data to this file. Prints a message indicating
    success or failure.

    Args:
        location (str): The directory path where the file should be saved.
        filename (str): The name of the file (e.g., "output.txt").
        data (str): The string data to be written to the file.
    """
    os.makedirs(location, exist_ok=True)  # Ensure the directory exists
    filepath = os.path.join(location, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(data)  # Write the raw string
        print(f"Data successfully saved to {filepath}")
    except Exception as e:
        print(f"Error saving file {filename}: {e}")


def extract_prompt(text: str, word: str) -> str:
    """Extracts and concatenates content from text blocks.

    For example, if `word` is "python", it will find all occurrences of
    ```python
    # some python code
    ```
    and return the content within these blocks.

    Args:
        text (str): The input string to search for code blocks.
        word (str): The keyword that identifies the type of code block
                    (e.g., "python", "latex", "bash").

    Returns:
        str: A single string containing all extracted content, concatenated
             by newlines and stripped of leading/trailing whitespace.
             Returns an empty string if no matching blocks are found.
    """
    # Escape the word if it contains special regex characters
    escaped_word = re.escape(word)
    code_block_pattern = rf"```{escaped_word}\s*\n(.*?)\n```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(block.strip() for block in code_blocks).strip()

    return extracted_code
