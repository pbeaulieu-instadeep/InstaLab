"""This module defines a framework for AI-driven generation.

Primarily in LaTeX format. It includes:
- An abstract `Command` class to define operations on a paper.
- Concrete command implementations for:
    - `Arxiv`: Searching and retrieving papers from arXiv.
    - `PaperReplace`: Completely replacing the LaTeX content of a paper.
    - `PaperEdit`: Editing specific line ranges within the LaTeX content.
- A `PaperSolver` class that orchestrates the paper generation process. It uses
  a language model (LLM) to decide on actions (commands) and applies them to
  iteratively refine the paper, aiming to improve its quality based on a scoring
  mechanism and predefined goals.
- Utility functions for LaTeX compilation and text extraction, imported from
  `utils` and `tools`.
"""

from __future__ import annotations

import os
import random
import re
import string
import time
from abc import abstractmethod
from copy import copy

from instalab.agents import get_score
from instalab.inference import query_model
from instalab.tools import ArxivSearch  # Assuming ArxivSearch is in tools.py
from instalab.utils import compile_latex  # Assuming compile_latex is in utils.py
from instalab.utils import extract_prompt  # Assuming extract_prompt is in utils.py


class Command:
    """Abstract base class for defining commands that can be executed.

    Subclasses must implement the abstract methods to define the specific
    behavior of a command, how it's identified, parsed, and executed.

    Attributes:
        cmd_type (str): A string identifier for the type of command.
                        Defaults to "OTHER".
    """

    def __init__(self):
        """Initializes the Command with a default command type."""
        self.cmd_type = "OTHER"

    @abstractmethod
    def docstring(self) -> str:
        """Provides a descriptive string for the command.

        This docstring is typically used to inform the language model about
        the command's purpose and usage.

        Returns:
            str: The documentation string for the command.
        """
        pass

    @abstractmethod
    def execute_command(self, *args) -> tuple:
        """Executes the command with the given arguments.

        Args:
            *args: Variable length argument list specific to the command.

        Returns:
            str | tuple: The result of the command execution. The type and
                         structure of the return value depend on the specific command.
        """
        pass

    @abstractmethod
    def matches_command(self, cmd_str: str) -> bool:
        """Checks if a given command string matches this command.

        Args:
            cmd_str (str): The command string to check.

        Returns:
            bool: True if the command string matches this command, False otherwise.
        """
        pass

    @abstractmethod
    def parse_command(self, cmd_str: str, *args) -> tuple:
        """Parses a command string to extract arguments or relevant information.

        Args:
            cmd_str (str): The command string to parse.
            *args: Additional args.

        Returns:
            tuple: Parsed arguments or information. The structure of the tuple
                   depends on the specific command.
        """
        pass


def execute_latex():
    """Placeholder function, possibly intended for LaTeX execution.

    Currently, it always returns True.

    Returns:
        bool: Always True.
    """
    return True


class Arxiv(Command):
    """Command for interacting with the arXiv API to search for papers and retrieve full text.

    It allows searching for paper summaries based on a query and fetching
    the full text of a paper given its arXiv ID.

    Attributes:
        arxiv_eng (ArxivSearch): An instance of the ArxivSearch tool.
        num_papers_per_search (int): Number of papers to retrieve per summary search.
        cmd_type (str): Identifier for this command type ("SEARCH-arxiv").
    """

    def __init__(self):
        """Initializes the Arxiv command with an ArxivSearch engine instance."""
        super().__init__()
        self.arxiv_eng = ArxivSearch()
        self.num_papers_per_search = 10
        self.cmd_type = "SEARCH-arxiv"

    def docstring(self) -> str:
        """Returns the documentation string for the Arxiv command.

        This string explains how to use the SUMMARY and FULL_TEXT sub-commands
        for searching arXiv.

        Returns:
            str: The help text for the Arxiv command.
        """
        return (
            "============= ARXIV SEARCH TOOL ============="
            "You also have access to machine learning paper from Arxiv. "
            "To search for summaries of papers on arxiv you can use the following command: ```SUMMARY\n<search query>\n```\n where <search query> is a string that will be used as the search query to find papers with semantically similar content and SUMMARY is just the word SUMMARY.\n"
            "To get the full paper text for an arXiv paper, use the following command: ```FULL_TEXT\n<arxiv paper id>\n```\n where <arxiv paper id> is the ID of the arXiv paper (which can be found by using the SUMMARY command), and FULL_TEXT is just the word FULL_TEXT. Make sure to read the full text using the FULL_TEXT command before adding it to your list of relevant papers.\n"
            "When you read arxiv paper, make sure to take note of the techniques they are using to solve their problem as well as the hyperparameters and implementation details. These are very important for successfully solving machine learning problems."
        )

    def execute_command(self, *args) -> tuple:
        """Executes an arXiv command (SUMMARY or FULL_TEXT).

        Args:
            *args: A tuple containing the sub-command type (args[0]) and
                   the query or paper ID (args[1]).

        Returns:
            str: The result from the arXiv search (summaries or full text).

        Raises:
            Exception: If an invalid Arxiv sub-command is provided.
        """
        # args[0] -> command type ("SUMMARY" or "FULL_TEXT")
        # args[1] -> query string or paper ID
        command_type = args[0]
        query_or_id = args[1]
        if command_type == "SUMMARY":
            return self.arxiv_eng.find_papers_by_str(query_or_id, self.num_papers_per_search)
        elif command_type == "FULL_TEXT":
            return self.arxiv_eng.retrieve_full_paper_text(query_or_id)
        raise Exception("Invalid Arxiv Search command type provided.")

    def matches_command(self, cmd_str: str) -> bool:
        """Checks if the command string indicates an Arxiv SUMMARY or FULL_TEXT command.

        Args:
            cmd_str (str): The command string from the LLM.

        Returns:
            bool: True if the string matches an Arxiv command, False otherwise.
        """
        if "```SUMMARY" in cmd_str:
            return True
        elif "```FULL_TEXT" in cmd_str:
            return True
        return False

    def parse_command(
        self, cmd_str: str, *args
    ) -> tuple[bool, tuple[str, list[str]]] | tuple[bool, None] | None:
        """Parses an Arxiv command string to extract the sub-command and its argument.

        Args:
            cmd_str (str): The command string from the LLM.
            *args: Additional arguments (unused in this implementation).

        Returns:
            tuple[bool, tuple[str, list[str]]] | tuple[bool, None] | None:
            If a SUMMARY command is found, returns `(True, ("SUMMARY", [query_lines]))`.
            If a FULL_TEXT command is found, returns `(True, ("FULL_TEXT", [id_lines]))`.
            If no Arxiv command is found, returns `(False, None)`.
            Returns `None` if parsing fails unexpectedly (though current logic aims for tuple[bool, ...]).
        """
        sum_text_lines = extract_prompt(cmd_str, "SUMMARY").split("\n")
        # Filter out empty strings that might result from split if extract_prompt returns ""
        sum_text_lines = [line for line in sum_text_lines if line.strip()]

        full_text_lines = extract_prompt(cmd_str, "FULL_TEXT").split("\n")
        full_text_lines = [line for line in full_text_lines if line.strip()]

        if not sum_text_lines and not full_text_lines:
            return False, None

        if sum_text_lines:
            # Assuming the query is the first non-empty line for simplicity
            return True, ("SUMMARY", sum_text_lines)

        if full_text_lines:
            # Assuming the paper ID is the first non-empty line
            return True, ("FULL_TEXT", full_text_lines)

        return None  # Should ideally not be reached if the above logic is exhaustive


class PaperReplace(Command):
    """Command to entirely replace the current LaTeX content of a paper.

    This command takes new LaTeX content and, after (optionally) compiling it
    to check for errors, replaces the existing paper content.

    Attributes:
        save_loc (str): The directory path where compilation artifacts are stored.
        cmd_type (str): Identifier for this command type ("PAPER-replace").
    """

    def __init__(self, save_loc: str):
        """Initializes the PaperReplace command.

        Args:
            save_loc (str): The directory path for saving compilation artifacts.
        """
        super().__init__()
        self.save_loc = save_loc
        self.cmd_type = "PAPER-replace"

    def docstring(self) -> str:
        """Returns the documentation string for the PaperReplace command.

        Explains how to use the REPLACE command to substitute the entire LaTeX document.

        Returns:
            str: The help text for the PaperReplace command.
        """
        return (
            "============= PAPER REPLACING TOOL =============\n"
            "You also have access to a paper replacing tool. \n"
            "This tool allows you to entirely re-write/replace all of the current latex and erase all existing latex.\n"
            "You can use this tool via the following command: ```REPLACE\n<latex here>\n```, where REPLACE is the word REPLACE and <latex here> will be the new latex that is replacing the entire set of old latex. This tool is useful if you want to make very significant changes, such as entirely changing the model, or the learning process. Before changing the existing latex to be your new latex, your new latex will be tested and if it returns an error it will not replace the existing latex. Try limiting the use of rewriting and aim for editing the latex more."
        )

    def execute_command(self, *args) -> tuple:
        """Executes the paper replacement.

        In this implementation, it simply returns the new LaTeX content.
        The actual replacement logic is handled by `PaperSolver.process_command`
        after successful parsing and compilation.

        Args:
            *args: A tuple where args[0] is expected to be the new LaTeX content
                   (as a list of lines or a single string).

        Returns:
            str: The new LaTeX content.
        """
        new_latex_content = args[0]
        if isinstance(new_latex_content, list):
            return "\n".join(new_latex_content)  # type: ignore
        return new_latex_content

    def matches_command(self, cmd_str: str) -> bool:
        """Checks if the command string indicates a PaperReplace command.

        Args:
            cmd_str (str): The command string from the LLM.

        Returns:
            bool: True if the string matches a PaperReplace command, False otherwise.
        """
        if "```REPLACE" in cmd_str:
            return True
        return False

    def parse_command(self, cmd_str: str, *args) -> tuple[bool, tuple[list[str] | None, str]]:
        """Parses a PaperReplace command string to extract the new LaTeX content.

        Args:
            cmd_str (str): The command string from the LLM.
            *args: A tuple where args[0] (compile_flag) is a boolean indicating
                   whether to compile the LaTeX.

        Returns:
            tuple[bool, tuple[list[str] | None, str]]:
            A tuple containing:
                - success (bool): True if parsing and compilation (if attempted)
                                  were successful (or no compilation error), False otherwise.
                - result_tuple (tuple):
                    - new_latex_lines (list[str] | None): List of new LaTeX lines if successful,
                                                       None otherwise.
                    - compilation_output (str): Output message from `compile_latex`.
        """
        compile_flag = args[0]  # Expecting compile_pdf boolean from PaperSolver
        new_latex = extract_prompt(cmd_str, "REPLACE")
        latex_ret = compile_latex(new_latex, self.save_loc, compile_tex=compile_flag)
        if "[CODE EXECUTION ERROR]" in latex_ret or "error" in latex_ret.lower():
            return False, (None, latex_ret)
        return True, (new_latex.split("\n"), latex_ret)


class PaperEdit(Command):
    """Command to edit specific line ranges within the current LaTeX content of a paper.

    It allows replacing a block of lines (from index N to M, inclusive) with
    new LaTeX lines. The changes are compiled (optionally) to check for errors.

    Attributes:
        save_loc (str): The directory path where compilation artifacts are stored.
        cmd_type (str): Identifier for this command type ("PAPER-edit").
    """

    def __init__(self, save_loc: str):
        """Initializes the PaperEdit command.

        Args:
            save_loc (str): The directory path for saving compilation artifacts.
        """
        super().__init__()
        self.save_loc = save_loc
        self.cmd_type = "PAPER-edit"

    def docstring(self) -> str:
        """Returns the documentation string for the PaperEdit command.

        Explains how to use the EDIT N M command to replace a range of lines
        in the LaTeX document.

        Returns:
            str: The help text for the PaperEdit command.
        """
        return (
            "============= PAPER EDITING TOOL =============\n"
            "You also have access to a paper editing tool. \n"
            "This tool allows you to replace lines indexed n through m (n:m) of the current latex with as many lines of new latex as you want to add. This removal is inclusive meaning that line n and m and everything between n and m is removed. This will be the primary way that you interact with latex. \n"
            "You can edit latex using the following command: ```EDIT N M\n<new lines to replace old lines>\n``` EDIT is the word EDIT, N is the first line index you want to replace and M the last line index you want to replace (everything between will also be removed), and <new lines to replace old lines> will be the new latex that is replacing the old latex. Before changing the existing latex to be your new latex, your new latex will be tested and if it returns an error it will not replace the existing latex. Your changes should significantly change the latex. You should write new paragraphs and update old ones. Try using the edit command often. Make sure to generate lots of text. You should also avoid editing lines 0 0, and should edit the main text of the paragraphs, such as editing lines in the middle of the text body."
        )

    def execute_command(self, *args) -> tuple[bool, list[str] | None, str]:
        """Executes the paper editing by replacing specified lines and compiling.

        Args:
            *args: A tuple containing:
                - edit_params (tuple): Contains (N, M, old_latex_lines, new_lines_to_add, compile_flag)
                    - N (int): Start line index for replacement.
                    - M (int): End line index for replacement.
                    - old_latex_lines (list[str]): The current LaTeX content as a list of lines.
                    - new_lines_to_add (list[str]): The new LaTeX lines to insert.
                    - compile_flag (bool): Whether to compile the modified LaTeX.

        Returns:
            tuple[bool, list[str] | None, str]:
                - success (bool): True if editing and compilation (if attempted)
                                  were successful, False otherwise.
                - modified_latex_lines (list[str] | None): The new list of LaTeX lines
                                                          if successful, None otherwise.
                - compilation_output (str): Output message from `compile_latex` or an error string.
        """
        try:
            edit_params = args[0]
            n, m, current_latex_lines_list, new_lines_to_add, compile_flag = edit_params

            # Ensure current_latex_lines_list is a mutable list of lines
            if isinstance(current_latex_lines_list, str):
                editable_latex_lines = current_latex_lines_list.splitlines()  # Or split('\n')
            else:  # Assuming it's already a list
                editable_latex_lines = list(current_latex_lines_list)  # Make a mutable copy

            # Validate N and M
            if not (0 <= n < len(editable_latex_lines) and n <= m < len(editable_latex_lines)):
                return (
                    False,
                    None,
                    f"Error: Line indices N({n}) or M({m}) are out of bounds for document length {len(editable_latex_lines)}.",
                )

            # Perform the edit: remove old lines, insert new lines
            # Remove lines from M down to N to avoid index shifting issues
            for i in range(m, n - 1, -1):
                editable_latex_lines.pop(i)

            # Insert new lines at position N
            for i, line_to_add in enumerate(new_lines_to_add):
                editable_latex_lines.insert(n + i, line_to_add)

            new_latex_str = "\n".join(editable_latex_lines)
            latex_ret = compile_latex(new_latex_str, self.save_loc, compile_tex=compile_flag)

            if "error" in latex_ret.lower() or "[CODE EXECUTION ERROR]" in latex_ret:
                return False, None, latex_ret
            return True, editable_latex_lines, latex_ret
        except Exception as e:
            return False, None, f"Error during paper edit execution: {str(e)}"

    def matches_command(self, cmd_str: str) -> bool:
        """Checks if the command string indicates a PaperEdit command.

        Args:
            cmd_str (str): The command string from the LLM.

        Returns:
            bool: True if the string matches a PaperEdit command, False otherwise.
        """
        if "```EDIT" in cmd_str:
            return True
        return False

    def parse_command(
        self, cmd_str: str, *args
    ) -> tuple[bool, tuple[int | None, int | None, list[str] | None, list[str] | None]]:
        """Parses a PaperEdit command string to extract line numbers and new content.

        Args:
            cmd_str (str): The command string from the LLM.
            *args: A tuple where args[0] is the current LaTeX content as a list of lines.

        Returns:
            tuple[bool, tuple[int | None, int | None, list[str] | None, list[str] | None]]:
            A tuple containing:
                - success (bool): True if parsing was successful, False otherwise.
                - parsed_args (tuple):
                    - N (int | None): Start line index.
                    - M (int | None): End line index.
                    - current_latex_lines (list[str] | None): The original LaTeX lines passed in.
                    - new_text_lines (list[str] | None): The new lines to insert.
                    All elements in `parsed_args` are None if parsing fails.
        """
        current_latex_lines = args[0]  # Expecting paper_lines from PaperSolver
        success = False
        n, m, new_text_lines = None, None, None
        try:
            extracted_content = extract_prompt(cmd_str, "EDIT")
            if not extracted_content:  # No content within ```EDIT ... ```
                return False, (None, None, None, None)

            lines = extracted_content.split("\n")
            if not lines:  # Should not happen if extracted_content is non-empty
                return False, (None, None, None, None)

            header_parts = lines[0].split()
            if len(header_parts) != 2:  # Expecting "N M"
                return False, (None, None, None, None)

            n_str, m_str = header_parts
            n = int(n_str)
            m = int(m_str)

            # Ensure N and M are non-negative and N <= M
            if n < 0 or m < 0 or n > m:
                return False, (None, None, None, None)

            new_text_lines = lines[1:]
            # It's valid to replace lines with empty content
            # if len(new_text_lines) == 0 and not lines[1:]: # If only "N M" and nothing else
            #    pass # Allow replacing with nothing

            success = True
            return success, (n, m, current_latex_lines, new_text_lines)
        except ValueError:  # Catches int conversion errors for N, M
            return False, (None, None, None, None)
        except Exception:  # Catch any other unexpected errors during parsing
            return False, (None, None, None, None)


# per_section_tips: A dictionary providing guidance for writing different
# sections of a scientific paper. Keys are section names (e.g., "abstract",
# "introduction"), and values are multi-line strings containing tips.
per_section_tips = {
    "abstract": """
- TL;DR of the paper
- What are we trying to do and why is it relevant?
- Why is this hard?
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- This must only be a single paragraph, not more.

Please make sure the abstract reads smoothly and is well-motivated. This should be one continuous paragraph with no breaks between the lines.
""",
    "introduction": """
- Longer version of the Abstract, i.e. of the entire paper
- What are we trying to do and why is it relevant?
- Why is this hard?
- How do we solve it (i.e. our contribution!)
- How do we verify that we solved it (e.g. Experiments and results)
- New trend: specifically list your contributions as bullet points
- Extra space? Future work!
""",
    "related work": """
- Academic siblings of our work, i.e. alternative attempts in literature at trying to solve the same problem.
- Goal is to “Compare and contrast” - how does their approach differ in either assumptions or method? If their method is applicable to our Problem Setting I expect a comparison in the experimental section. If not, there needs to be a clear statement why a given method is not applicable.
- Note: Just describing what another paper is doing is not enough. We need to compare and contrast.
""",
    "background": """
- Academic Ancestors of our work, i.e. all concepts and prior work that are required for understanding our method.
- Usually includes a subsection, Problem Setting, which formally introduces the problem setting and notation (Formalism) for our method. Highlights any specific assumptions that are made that are unusual.
- Make sure to use mathematical notation when necessary.
- Note: If our paper introduces a novel problem setting as part of its contributions, it's best to have a separate Section.
""",
    "methods": """
- What we do. Why we do it. All described using the general Formalism introduced in the Problem Setting and building on top of the concepts / foundations introduced in Background.
- Make sure you clearly report precise mathematical equations in the methods section and the precise methodology.
""",
    "experimental setup": """
- How do we test that our stuff works? Introduces a specific instantiation of the Problem Setting and specific implementation details of our Method for this Problem Setting.
- Do not imagine unknown hardware details.
- Includes a description of the dataset, evaluation metrics, important hyperparameters, and implementation details.
""",
    "results": """
- Shows the results of running Method on our problem described in Experimental Setup.
- Includes statements on hyperparameters and other potential issues of fairness.
- Only includes results that have actually been run and saved in the logs. Do not hallucinate results that don't exist.
- Make sure you clearly and numerically report experimental results in the results section.
- If results exist: compares to baselines and includes statistics and confidence intervals.
- If results exist: includes ablation studies to show that specific parts of the method are relevant.
- Discusses limitations of the method.
- Make sure to include all the results from the experiments, and include all relevant figures.
""",
    "discussion": """
- Brief recap of the entire paper.
- To keep going with the analogy, you can think of future work as (potential) academic offspring.
""",
}


class PaperSolver:
    """Orchestrates the AI-driven generation and iterative refinement of a scientific paper.

    This class uses a language model (LLM) to generate commands for creating or
    modifying LaTeX paper content. It manages the paper's state, applies commands,
    compiles LaTeX, and potentially scores the paper to guide the LLM towards
    improving the document.

    Attributes:
        notes (list | str | None): Notes or instructions for the LLM.
        max_steps (int): Maximum number of refinement steps in the `solve` loop.
        plan (str | None): The research plan or outline for the paper.
        lit_review (str | None): Literature review content.
        ref_papers (str | None): Content of reference papers.
        topic (str | None): The main topic of the paper.
        gemini_api_key (str | None): API key for Gemini LLM, if used.
        compile_pdf (bool): Flag to control PDF compilation after edits.
        save_loc (str | None): Directory to save compilation artifacts and outputs.
        supress_print (bool): If True, suppresses most print statements.
        max_papers (int): Maximum number of best paper versions to keep.
        st_hist_len (int): Short-term history length (purpose not fully clear from context).
        min_gen_trials (int): Minimum generation trials in the `solve` loop.
        paper_lines (list[str] | str): Current content of the paper, as a list of lines or a single string.
        prev_paper_ret (str): Output from the last LaTeX compilation attempt.
        section_related_work (dict): Stores related work found for different sections.
        best_report (list): Stores tuples of (paper_lines, score, compilation_output) for best papers.
        commands (list[Command]): List of available command objects.
        model (str): String representation of the LLM being used.
        prev_working_report (list[str]): A copy of the last known good paper state.
    """

    def __init__(
        self,
        notes: list | str | None = None,
        max_steps: int = 10,
        plan: str | None = None,
        lit_review: str | None = None,
        ref_papers: str | None = None,
        topic: str | None = None,
        gemini_api_key: str | None = None,
        compile_pdf: bool = True,
        save_loc: str | None = None,
    ):
        """Initializes the PaperSolver.

        Args:
            notes (list | str | None, optional): Notes/instructions for the LLM. Defaults to None.
            max_steps (int, optional): Max refinement steps. Defaults to 10.
            plan (str | None, optional): Research plan. Defaults to None.
            lit_review (str | None, optional): Literature review. Defaults to None.
            ref_papers (str | None, optional): Reference papers. Defaults to None.
            topic (str | None, optional): Paper topic. Defaults to None.
            gemini_api_key (str | None, optional): Gemini API key. Defaults to None.
            compile_pdf (bool, optional): Whether to compile PDF after edits. Defaults to True.
            save_loc (str | None, optional): Directory for saving outputs. Defaults to a generated name if None.
        """
        self.supress_print = True  # Default as per original code
        self.notes = notes if notes is not None else []
        self.plan = plan if plan is not None else ""
        self.lit_review = lit_review if lit_review is not None else ""
        self.ref_papers = ref_papers if ref_papers is not None else ""
        self.topic = topic if topic is not None else ""

        if save_loc is None:
            # Create a default save location if not provided
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            random_suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
            self.save_loc = f"paper_solver_output_{timestamp}_{random_suffix}"
        else:
            self.save_loc = save_loc
        os.makedirs(self.save_loc, exist_ok=True)

        self.compile_pdf = compile_pdf
        # self.notes = notes # Already assigned
        self.max_papers = 1
        self.st_hist_len = 10
        self.min_gen_trials = 2
        self.max_steps = max_steps
        self.paper_lines: list[str] = []  # Initialize as empty list of strings
        self.prev_paper_ret = ""
        self.section_related_work: dict = {}
        self.gemini_api_key = gemini_api_key

        # Initialize attributes that are set later
        self.best_score: float | None = None
        self.commands: list[Command] = []
        self.best_report: list[tuple[list[str], float, str]] = []
        self.prev_working_report: list[str] = []

    def solve(self) -> tuple[str, str]:
        """Runs the iterative paper refinement loop.

        In each step, it queries the LLM for a command, processes the command,
        updates the paper, and scores it. It keeps track of the best paper
        version found.

        Returns:
            tuple[str, str]:
                - model_response (str): The last model response that led to the best paper.
                - command_string (str): The command string corresponding to that model response.
        """
        num_attempts = 0
        best_pkg: tuple[list[str], str, str, str] | None = None
        top_score: float | None = None
        # self.prev_paper_ret should be initialized (e.g. from initial_solve or "" )

        # Ensure self.best_report is initialized and has at least one entry to pick from
        if not self.best_report or not self.best_report[0][0]:
            # Fallback if best_report is empty or malformed after initial_solve
            if self.paper_lines:  # if initial solve populated paper_lines directly
                current_paper_lines_list = (
                    self.paper_lines
                    if isinstance(self.paper_lines, list)
                    else self.paper_lines.splitlines()
                )
            else:
                current_paper_lines_list = [
                    "% Placeholder if no initial paper exists"
                ]  # Default if truly empty
            print(
                "Warning: best_report was empty or invalid at the start of solve(). Using current paper_lines or placeholder."
            )
        else:
            current_paper_lines_list = copy(random.choice(self.best_report)[0])

        for current_step in range(self.max_steps):
            if not self.supress_print:
                print(f"\n--- Solve Step {current_step + 1}/{self.max_steps} ---")

            # Select a paper to work on (e.g., from the best ones found so far)
            if self.best_report:
                self.paper_lines = copy(random.choice(self.best_report)[0])
            else:  # Fallback if best_report is somehow empty
                self.paper_lines = current_paper_lines_list

            model_resp = query_model(
                system_prompt=self.system_prompt(),
                prompt="\nNow please enter a command: ",
                gemini_api_key=self.gemini_api_key,
            )
            model_resp = self.clean_text(model_resp)

            cmd_str_result, paper_lines_result, prev_paper_ret_result, score_result = (
                self.process_command(model_resp)
            )

            if score_result is not None:
                if top_score is None or score_result > top_score:
                    best_pkg = (
                        copy(paper_lines_result),
                        copy(prev_paper_ret_result),
                        copy(model_resp),
                        copy(cmd_str_result if cmd_str_result is not None else "N/A"),
                    )
                    top_score = score_result
                    if not self.supress_print:
                        print(f"$$$ New best score: {top_score}")

            num_attempts += 1
            if num_attempts >= self.min_gen_trials and top_score is not None:
                # Condition to potentially break early if enough trials and a good score
                # This logic might need refinement based on desired behavior
                pass  # Continue for max_steps unless specific break conditions are met

            if not self.supress_print:
                print(
                    f"@@@ Command Exec // Step {current_step + 1} Attempt {num_attempts}: ",
                    str(cmd_str_result if cmd_str_result else "No valid command processed").replace(
                        "\n", " | "
                    ),
                )
                print(f"$$$ Current Score: {score_result if score_result is not None else 'N/A'}")

        if best_pkg is None:
            # Handle case where no successful command execution leading to a score occurred
            print("Warning: No best package found during solve loop. Returning placeholder.")
            return "No successful operation", "No command executed"

        # Unpack best_pkg
        final_paper_lines, final_prev_paper_ret, final_model_resp, final_cmd_str = best_pkg

        # Update the best_report with the top-scoring paper from this solve loop
        if top_score is not None and (not self.best_report or top_score > self.best_report[-1][1]):
            if len(self.best_report) >= self.max_papers and self.best_report:
                self.best_report.pop(-1)  # Remove the lowest scoring one if full
            self.best_report.append((copy(final_paper_lines), top_score, final_prev_paper_ret))
            self.best_report.sort(key=lambda x: x[1], reverse=True)

        # Update main paper state to the best one found
        self.paper_lines = final_paper_lines
        self.prev_paper_ret = final_prev_paper_ret

        return final_model_resp, final_cmd_str

    def initial_solve(self) -> None:
        """Initializes the solver by generating an initial version of the paper.

        It sets up the necessary commands and calls `gen_initial_report` to create
        the first draft of the paper scaffold and content. The result is stored
        as the initial best report.
        """
        self.best_score = None  # Initialize best_score for this session
        self.commands = [PaperReplace(self.save_loc)]  # Initially, only REPLACE is available

        if not self.supress_print:
            print("--- Starting Initial Paper Generation ---")

        # gen_initial_report returns (latex_lines_list, prev_latex_ret_str, score_float_or_none)
        init_report_lines, init_return_str, init_score = self.gen_initial_report()

        if init_score is None:  # If initial generation failed to produce a score
            print("Warning: Initial paper generation did not yield a score. Using 0.0 as fallback.")
            init_score = 0.0  # Fallback score
            if not init_report_lines:  # If no lines were generated
                init_report_lines = ["% Initial generation failed to produce content."]

        self.best_score = init_score
        # Ensure best_report stores list of lines, score, and compilation output
        self.best_report = [
            (copy(init_report_lines), self.best_score, init_return_str)
            for _ in range(self.max_papers)
        ]  # Replicate for initial pool

        self.paper_lines = copy(init_report_lines)  # Set current paper_lines
        self.prev_paper_ret = init_return_str  # Set current compilation return

        # After initial generation, switch to allowing edits
        self.commands = [PaperEdit(self.save_loc), PaperReplace(self.save_loc)]  # Add PaperEdit
        self.prev_working_report = copy(self.paper_lines)

        if not self.supress_print:
            print("--- Initial Paper Generation Complete ---")
            if self.best_score is not None:
                print(f"Initial Score: {self.best_score}")

    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans the input text.

        This is a simple utility to normalize code block formatting that might
        come from the LLM.

        Args:
            text (str): The text to clean.

        Returns:
            str: The cleaned text.
        """
        text = text.replace("```\n", "```")
        return text

    def gen_initial_report(self) -> tuple[list[str], str, float | None]:  # noqa: C901
        """Generates the initial scaffold and content for the paper, section by section.

        It iterates through predefined sections, prompting the LLM to generate
        LaTeX for each. For some sections, it may perform an arXiv search to
        provide related papers to the LLM for citation.

        Returns:
            tuple[list[str], str, float | None]:
                - paper_content_lines (list[str]): The LaTeX content of the generated paper as a list of lines.
                - compilation_output (str): The output from the final LaTeX compilation.
                - score (float | None): The score of the initial paper, if calculable.
        """
        num_total_attempts = 0  # Total attempts across all sections
        arx = ArxivSearch()
        current_paper_scaffold_str = ""
        final_compilation_output = "No compilation attempted for initial report."
        final_score: float | None = None

        # Define the sections in order
        sections_to_generate = [
            "scaffold",
            "abstract",
            "introduction",
            "related work",
            "background",
            "methods",
            "experimental setup",
            "results",
            "discussion",
        ]

        for section_name in sections_to_generate:
            section_generation_complete = False
            num_section_attempts = 0
            max_section_attempts = 5  # Max attempts per section

            if not self.supress_print:
                print(f"\n--- Generating section: {section_name.upper()} ---")

            # For relevant sections, find related papers
            if section_name in [
                "introduction",
                "related work",
                "background",
                "methods",
                "discussion",
            ]:
                papers_found = ""
                search_attempts = 0
                max_search_attempts = 3
                while not papers_found and search_attempts < max_search_attempts:
                    search_prompt_detail = ""
                    if search_attempts > 0:
                        search_prompt_detail = " Previous attempt failed or found no papers. Please try a different or simpler search query."

                    search_query = query_model(
                        prompt=f"Given the research topic: '{self.topic}' and research plan: \n{self.plan}\n\nPlease provide a concise search query for arXiv to find relevant papers for the '{section_name}' section. Respond only with the search query string.{search_prompt_detail}",
                        system_prompt=f"You are an expert research assistant helping find relevant literature for a scientific paper. The current section is '{section_name}'. The query should be a simple string.",
                        gemini_api_key=self.gemini_api_key,
                    )
                    search_query = search_query.replace('"', "").strip()
                    if search_query:  # Only search if a query was generated
                        if not self.supress_print:
                            print(
                                f"Searching arXiv for '{section_name}' with query: '{search_query}'"
                            )
                        papers_found = arx.find_papers_by_str(
                            query=search_query, n=5
                        )  # Fetch fewer for context
                        if papers_found and papers_found != "No papers found matching the query.":
                            self.section_related_work[section_name] = papers_found
                            if not self.supress_print:
                                print(f"Found related papers for {section_name}.")
                        else:
                            if not self.supress_print:
                                print(f"No papers found or error for query: '{search_query}'")
                            papers_found = ""  # Reset if no meaningful result
                    search_attempts += 1
                    time.sleep(1)  # Be polite to API

            while not section_generation_complete and num_section_attempts < max_section_attempts:
                num_section_attempts += 1
                num_total_attempts += 1
                temp_scaffold_for_this_attempt = copy(current_paper_scaffold_str)
                error_context = ""

                # Construct the prompt for the LLM
                llm_prompt_suffix = ""
                if section_name == "scaffold":
                    llm_prompt_suffix = f"{error_context}\nNow please enter the ```REPLACE command to create the entire paper scaffold. The scaffold should include placeholders like [ABSTRACT HERE], [INTRODUCTION HERE], etc., for all standard sections. The title should be based on the topic: '{self.topic}'."
                else:
                    related_papers_context = ""
                    if section_name in self.section_related_work:
                        related_papers_context = f"Here are some related papers you might find useful for context or citation (e.g., (arXiv:1234.56789)): {self.section_related_work[section_name]}\n"
                    llm_prompt_suffix = f"{error_context}\n{related_papers_context}\nNow please enter the ```REPLACE command to generate ONLY the content for the '{section_name.upper()}' section. Do not include LaTeX documentclass, usepackage, or \\section commands. Just provide the body text, equations, tables, etc., for this section. This content will replace the '[{section_name.upper()} HERE]' placeholder."

                # Query the LLM
                model_resp_str = query_model(
                    system_prompt=self.system_prompt(
                        section=section_name
                    ),  # System prompt tailored to the section
                    prompt=llm_prompt_suffix,
                    gemini_api_key=self.gemini_api_key,
                )
                model_resp_str = self.clean_text(model_resp_str)

                # Process the LLM's response
                if section_name == "scaffold":
                    # For scaffold, the LLM's ```REPLACE contains the full scaffold
                    # We expect placeholders like [ABSTRACT HERE]
                    is_valid_scaffold = True
                    for placeholder in [
                        "[ABSTRACT HERE]",
                        "[INTRODUCTION HERE]",
                        "[METHODS HERE]",
                        "[RESULTS HERE]",
                        "[DISCUSSION HERE]",
                    ]:
                        if placeholder not in extract_prompt(model_resp_str, "REPLACE"):
                            is_valid_scaffold = False
                            cmd_str_result = f"Error: Scaffold missing placeholder: {placeholder}."
                            break
                    if not is_valid_scaffold:
                        if not self.supress_print:
                            print(f"@@@ INIT ATTEMPT (Scaffold): {cmd_str_result}")
                        continue  # Try generating scaffold again

                    # If scaffold is valid, process it (compilation is effectively for the scaffold itself)
                    (
                        cmd_str_result,
                        paper_content_lines_list,
                        final_compilation_output,
                        score_result,
                    ) = self.process_command(
                        model_resp_str,
                        scoring=False,  # No scoring for intermediate scaffold
                    )
                else:  # For other sections, LLM provides content for one section
                    section_content_from_llm = extract_prompt(model_resp_str, "REPLACE")
                    if (
                        "documentclass{article}" in section_content_from_llm
                        or "usepackage{" in section_content_from_llm
                        or "\\section{" in section_content_from_llm
                    ):
                        cmd_str_result = "Error: Section content included forbidden LaTeX commands (documentclass, usepackage, section)."
                        if not self.supress_print:
                            print(f"@@@ INIT ATTEMPT ({section_name}): {cmd_str_result}")
                        continue  # Try generating this section again

                    # Replace placeholder in the current full scaffold
                    placeholder_tag = f"[{section_name.upper()} HERE]"
                    if placeholder_tag not in temp_scaffold_for_this_attempt:
                        cmd_str_result = f"Error: Placeholder {placeholder_tag} not found in the current scaffold. Cannot insert section content."
                        if not self.supress_print:
                            print(f"@@@ INIT ATTEMPT ({section_name}): {cmd_str_result}")
                        # This might indicate a problem with the scaffold itself or a wrong section name
                        # For simplicity, we'll try generating the section again, but this could loop if scaffold is bad
                        continue

                    temp_scaffold_for_this_attempt = temp_scaffold_for_this_attempt.replace(
                        placeholder_tag, section_content_from_llm
                    )
                    # Create a new ```REPLACE command with the updated full scaffold for processing
                    full_paper_model_resp = (
                        "```REPLACE\n" + temp_scaffold_for_this_attempt + "\n```"
                    )
                    (
                        cmd_str_result,
                        paper_content_lines_list,
                        final_compilation_output,
                        score_result,
                    ) = self.process_command(
                        full_paper_model_resp,
                        scoring=(
                            section_name == sections_to_generate[-1]
                        ),  # Score only on the last section
                    )

                if not self.supress_print:
                    print(
                        f"@@@ INIT ATTEMPT ({section_name} - attempt {num_section_attempts}): ",
                        str(cmd_str_result if cmd_str_result else "No command processed").replace(
                            "\n", " | "
                        ),
                    )

                if paper_content_lines_list is not None and not (
                    "FAILED" in (cmd_str_result or "") or "Error:" in (cmd_str_result or "")
                ):
                    section_generation_complete = True
                    current_paper_scaffold_str = "\n".join(
                        paper_content_lines_list
                    )  # Update the main scaffold string
                    self.paper_lines = (
                        current_paper_scaffold_str.splitlines()
                    )  # Update solver's current paper state
                    if score_result is not None:
                        final_score = (
                            score_result  # Capture score if available (esp. for last section)
                        )
                    if not self.supress_print:
                        print(f"$$$ Section '{section_name}' generated successfully.")
                else:
                    if not self.supress_print:
                        print(
                            f"--- Section '{section_name}' generation attempt {num_section_attempts} failed. Retrying..."
                        )

            if not section_generation_complete:
                if not self.supress_print:
                    print(
                        f"!!! Failed to generate section '{section_name}' after {max_section_attempts} attempts. Moving on."
                    )
                # If a section fails, we might have an incomplete paper.
                # For scaffold, this is critical. For others, we might proceed with a missing section.
                if section_name == "scaffold":
                    print(
                        "CRITICAL: Failed to generate paper scaffold. Aborting initial generation."
                    )
                    return ["% Scaffold generation failed."], "Scaffold generation failed.", None

        if not self.supress_print:
            print("\n$" * 10, " INITIAL PAPER GENERATION PROCESS COMPLETE ", "$" * 10)

        # Ensure paper_lines is a list of strings
        final_paper_lines_list = current_paper_scaffold_str.splitlines()
        self.paper_lines = final_paper_lines_list  # Final update to solver's state

        # Attempt a final score if not already scored on the last section
        if final_score is None and final_paper_lines_list:
            try:
                final_score, score_cmd_str, is_valid = get_score(
                    self.plan, "\n".join(final_paper_lines_list)
                )
                if not is_valid:
                    final_score = None  # Invalidate score if paper is not valid by scoring criteria
                if not self.supress_print:
                    print(
                        f"Final score for initial report: {final_score}, Message: {score_cmd_str}"
                    )
            except Exception as e:
                if not self.supress_print:
                    print(f"Error during final scoring of initial report: {e}")
                final_score = None

        return final_paper_lines_list, final_compilation_output, final_score

    def process_command(  # noqa: C901
        self, model_resp: str, scoring: bool = True
    ) -> tuple[str | None, list[str], str, float | None]:
        """Processes a command string received from the LLM.

        It identifies the command, parses it, executes it, and optionally scores
        the resulting paper. Handles `PAPER-edit` and `PAPER-replace` commands.

        Args:
            model_resp (str): The raw command string from the LLM.
            scoring (bool, optional): Whether to score the paper after the command.
                                      Defaults to True.

        Returns:
            tuple[str | None, list[str], str, float | None]:
                - cmd_status_msg (str | None): A message describing the outcome of the command processing.
                - resulting_paper_lines (list[str]): The state of the paper lines after the command.
                - compilation_output (str): Output from LaTeX compilation.
                - score (float | None): The score of the paper after the command, if scoring was done.
        """
        cmd_status_msg: str | None = "No matching command found."
        score: float | None = None
        # Ensure self.paper_lines is a list. If it's a string, split it.
        # Make a copy to avoid modifying the original if the command fails.
        current_paper_lines_list = (
            list(self.paper_lines)
            if isinstance(self.paper_lines, list)
            else self.paper_lines.splitlines()
        )
        resulting_paper_lines_list = copy(current_paper_lines_list)
        compilation_output_str = self.prev_paper_ret  # Default to previous if no new compilation

        # Handle figure path replacement globally if these figures are expected to exist
        # This might be better placed where figures are actually generated or confirmed.
        # For now, keeping it as per original logic.
        # CWD_PATH = os.getcwd().replace("\\", "/") # Ensure consistent path separators
        # model_resp = model_resp.replace("Figure_1.png", f"{CWD_PATH}/Figure_1.png")
        # model_resp = model_resp.replace("Figure_2.png", f"{CWD_PATH}/Figure_2.png")

        for cmd_obj in self.commands:
            if cmd_obj.matches_command(model_resp):
                if cmd_obj.cmd_type == "PAPER-edit":
                    parse_success, parsed_args = cmd_obj.parse_command(
                        model_resp, resulting_paper_lines_list
                    )  # Pass current lines
                    if parse_success:
                        n, m, _, new_text = (
                            parsed_args  # original_latex_lines is part of parsed_args but we use current paper_lines
                        )

                        # Args for execute_command: (N, M, current_latex_lines, new_lines_to_add, compile_flag)
                        exec_success, exec_result_lines, compilation_output_str = (
                            cmd_obj.execute_command(
                                (n, m, resulting_paper_lines_list, new_text, self.compile_pdf)
                            )
                        )

                        if exec_success and exec_result_lines is not None:
                            resulting_paper_lines_list = (
                                exec_result_lines  # Update with successfully edited lines
                            )
                            if scoring:
                                score, score_msg, is_valid = get_score(
                                    self.plan, "\n".join(resulting_paper_lines_list)
                                )
                                if not is_valid:
                                    score = None  # Invalidate if scoring says so
                            else:
                                score, score_msg, is_valid = 0.0, "Scoring skipped.", True

                            if is_valid:  # and no compilation error indicated by exec_success
                                cmd_status_msg = f"Paper successfully edited. {score_msg}"
                                self.prev_working_report = copy(
                                    resulting_paper_lines_list
                                )  # Update last good state
                            else:
                                cmd_status_msg = f"Paper edit applied, but validation/scoring failed: {score_msg}. Compilation: {compilation_output_str}"
                                # Revert to previous working state if scoring/validation fails significantly
                                resulting_paper_lines_list = copy(self.prev_working_report)
                                score = None  # Ensure score is None if reverted or invalid
                        else:  # Execution failed (e.g., compilation error from execute_command)
                            cmd_status_msg = (
                                f"Paper edit FAILED: {compilation_output_str}. Paper reverted."
                            )
                            resulting_paper_lines_list = copy(self.prev_working_report)  # Revert
                            score = None
                    else:  # Parsing failed
                        cmd_status_msg = "Paper edit command parsing FAILED. No changes made."
                        # resulting_paper_lines_list remains copy(current_paper_lines_list) or prev_working_report
                        score = None

                    if not self.supress_print:
                        print(f"$$$$ {cmd_status_msg} $$$$")
                    return cmd_status_msg, resulting_paper_lines_list, compilation_output_str, score

                elif cmd_obj.cmd_type == "PAPER-replace":
                    # parse_command for REPLACE takes (cmd_str, compile_flag)
                    parse_success, parsed_result = cmd_obj.parse_command(
                        model_resp, self.compile_pdf
                    )
                    if parse_success:
                        new_latex_lines_list, compilation_output_str = (
                            parsed_result  # (list[str] | None, str)
                        )
                        if (
                            new_latex_lines_list is not None
                        ):  # Indicates successful parse and initial compile check
                            # execute_command for REPLACE in this setup just returns the content
                            # The actual "replacement" happens here by updating resulting_paper_lines_list
                            resulting_paper_lines_list = new_latex_lines_list

                            if scoring:
                                score, score_msg, is_valid = get_score(
                                    self.plan, "\n".join(resulting_paper_lines_list)
                                )
                                if not is_valid:
                                    score = None
                            else:
                                score, score_msg, is_valid = 0.0, "Scoring skipped.", True

                            if is_valid:
                                cmd_status_msg = f"Paper successfully replaced. {score_msg}"
                                self.prev_working_report = copy(resulting_paper_lines_list)
                            else:
                                cmd_status_msg = f"Paper replacement applied, but validation/scoring failed: {score_msg}. Compilation: {compilation_output_str}"
                                resulting_paper_lines_list = copy(self.prev_working_report)
                                score = None
                        else:  # new_latex_lines_list is None, meaning compilation error during parse
                            cmd_status_msg = f"Paper replacement FAILED due to compilation error: {compilation_output_str}. Paper reverted."
                            resulting_paper_lines_list = copy(self.prev_working_report)
                            score = None
                    else:  # Parsing / initial compilation check failed
                        _, compilation_output_str = parsed_result  # Get the error message
                        cmd_status_msg = f"Paper replacement command parsing/compilation FAILED: {compilation_output_str}. No changes made."
                        # resulting_paper_lines_list remains as it was
                        score = None

                    if not self.supress_print:
                        print(f"$$$$ {cmd_status_msg} $$$$")
                    return cmd_status_msg, resulting_paper_lines_list, compilation_output_str, score

        # If no command matched and was processed by the loop:
        return cmd_status_msg, resulting_paper_lines_list, compilation_output_str, score

    def generate_paper_lines(self, code_lines: list[str]) -> str:
        """Formats a list of LaTeX code lines with line numbers for display.

        Args:
            code_lines (list[str]): A list of strings, where each string is a line of LaTeX code.

        Returns:
            str: A single string with each original line prefixed by its line number
                 (0-indexed) and a pipe symbol, and ending with a newline.
        """
        codestr = ""
        for index, line_content in enumerate(code_lines):
            codestr += f"{index} |{line_content}\n"
        return codestr

    def system_prompt(self, commands: bool = True, section: str | None = None) -> str:
        """Generates a system prompt for the LLM based on the current state and task.

        The prompt includes role descriptions, task instructions, notes, literature review,
        the research plan, current paper content, available commands, and section-specific
        guidance if applicable.

        Args:
            commands (bool, optional): Whether to include command descriptions in the prompt.
                                       Defaults to True.
            section (str | None, optional): The specific section being worked on, if any.
                                            This influences section-specific tips. Defaults to None.

        Returns:
            str: The constructed system prompt for the LLM.
        """
        length_guidance = ""
        if section == "abstract":
            length_guidance = "This section should be ONLY 1 paragraph."
        elif section is not None:  # For other sections being generated initially
            length_guidance = "This section should be approximately 2-4 paragraphs, so your output should be several paragraphs of LaTeX."

        methods_figure_instructions = ""
        if section == "methods":
            fig1_text = """
\\begin{figure}[h]
\\caption{<caption here>}
\\centering
\\includegraphics[width=\\textwidth]{Figure_1.png}
\\label{fig:fig1}
\\end{figure}
"""
            fig2_text = """
\\begin{figure}[h]
\\caption{<caption here>}
\\centering
\\includegraphics[width=\\textwidth]{Figure_2.png}
\\label{fig:fig2} % Corrected label for fig2
\\end{figure}
"""
            # Check for figures relative to the save_loc or a known figures directory
            # Assuming figures might be in self.save_loc or a subdirectory thereof
            # For simplicity, using relative paths as in original, assuming pdflatex cwd is set appropriately
            # This part needs to align with how figures are actually managed and where pdflatex runs from
            fig1_exists = os.path.exists(
                os.path.join(self.save_loc, "Figure_1.png")
            ) or os.path.exists("Figure_1.png")
            fig2_exists = os.path.exists(
                os.path.join(self.save_loc, "Figure_2.png")
            ) or os.path.exists("Figure_2.png")

            if fig1_exists and fig2_exists:
                methods_figure_instructions = f"You ABSOLUTELY must include Figure_1.png and Figure_2.png in your paper using the following LaTeX structures on new lines. Place them in suitable, separate locations within the methods section:\nFigure 1:\n{fig1_text}\nFigure 2:\n{fig2_text}"
            elif fig1_exists:
                methods_figure_instructions = f"You ABSOLUTELY must include Figure_1.png in your paper using the following LaTeX structure on a new line:\n{fig1_text}"
            elif fig2_exists:
                methods_figure_instructions = f"You ABSOLUTELY must include Figure_2.png in your paper using the following LaTeX structure on a new line:\n{fig2_text}"

        section_specific_cmd_guidance = ""
        if section == "scaffold":
            section_specific_cmd_guidance = (
                "Your objective right now is to ONLY build the scaffolding for the paper. "
                "Do not include any text in the body of the paper, but create an empty scaffold "
                "with placeholders for each section: [ABSTRACT HERE], [INTRODUCTION HERE], "
                "[RELATED WORK HERE], [BACKGROUND HERE], [METHODS HERE], [EXPERIMENTAL SETUP HERE], "
                "[RESULTS HERE], and [DISCUSSION HERE]. "
                f"The paper's title should be 'Research Report: <A title you choose based on the topic: {self.topic}>'. "
                "For the author, write 'Agent Laboratory'. Ensure the output is compilable LaTeX."
            )
        elif section is not None and section in per_section_tips:
            section_specific_cmd_guidance = (
                f"Your ONLY goal is to generate LaTeX content for the '{section.upper()}' section. "
                "DO NOT INCLUDE any \\documentclass, \\usepackage, \\title, \\author, \\date, or \\section commands. "
                f"ONLY provide the body text for this specific section. {length_guidance} "
                "Use mathematical equations, numbers, and tables where appropriate. "
                "Remember that to include a percentage sign %, you must use \\% to avoid it being treated as a comment. "
                f"Here are some tips for writing this section:\n{per_section_tips[section]}\n"
                f"{methods_figure_instructions}\n"
            )

        # Calculate current paper length (approximate words)
        current_paper_text = "".join(self.paper_lines)
        # A simple word count: split by space, filter out non-alphanumeric, then count
        words_in_paper = [
            word
            for word in re.split(r"\s+", current_paper_text)
            if word.strip(string.punctuation).isalnum()
        ]
        paper_len_words = len(words_in_paper)

        target_paper_length_words = 4000
        paper_progress_msg = ""
        if paper_len_words < target_paper_length_words:
            paper_progress_msg = f"The current paper is approximately {paper_len_words} words long. You need to add about {target_paper_length_words - paper_len_words} more words to reach the target of {target_paper_length_words} words. Focus on expanding the content significantly."
        else:
            paper_progress_msg = f"The current paper is approximately {paper_len_words} words long, which meets or exceeds the target of {target_paper_length_words} words. Focus on refining and improving the quality."

        if not self.supress_print:
            print(paper_progress_msg)  # Print progress for operator

        command_descriptions_str = self.command_descriptions() if commands else ""

        reference_papers_str = ""
        if isinstance(self.ref_papers, list) and self.ref_papers:  # If it's a list of paper strings
            reference_papers_str = "\n\n---\n\n".join(self.ref_papers)
        elif isinstance(self.ref_papers, str) and self.ref_papers:  # If it's a single string
            reference_papers_str = self.ref_papers

        if reference_papers_str:
            reference_papers_str = f"Here is a high-quality reference paper (or papers) you can draw inspiration from for style and structure:\n{reference_papers_str}\n\n"

        # Ensure self.paper_lines is a list for generate_paper_lines
        paper_lines_list = (
            self.paper_lines
            if isinstance(self.paper_lines, list)
            else str(self.paper_lines).splitlines()
        )
        current_paper_formatted_str = self.generate_paper_lines(paper_lines_list)

        # Truncate long parts of the prompt if necessary to avoid excessive length
        lit_review_display = str(self.lit_review)[:10000] + (
            "..." if len(str(self.lit_review)) > 10000 else ""
        )
        plan_display = str(self.plan)[:5000] + ("..." if len(str(self.plan)) > 5000 else "")
        notes_display = str(self.notes)[:5000] + ("..." if len(str(self.notes)) > 5000 else "")

        prompt = (
            f"{reference_papers_str}"
            f"{self.role_description()}\n\n"
            f"TASK INSTRUCTIONS:\n{self.phase_prompt()}\n\n"
            f"NOTES & GENERAL TIPS:\n{notes_display}\n\n"
            f"LITERATURE REVIEW PROVIDED:\n{lit_review_display}\n\n"
            f"ORIGINAL RESEARCH PLAN:\n{plan_display}\n\n"
            f"YOUR GOAL: Write a high-quality research paper. Aim for a score maximization. "
            f"The paper should be comprehensive, targeting ~8 pages or {target_paper_length_words} words. "
            f"{paper_progress_msg}\n\n"
            f"{command_descriptions_str}\n\n"
            f"CURRENT PAPER STATE:\n{current_paper_formatted_str}\n\n"
            f"{section_specific_cmd_guidance}"
        )
        return prompt

    def command_descriptions(self) -> str:
        """Generates a string describing all available commands and their usage.

        This is used to inform the LLM about the tools it can use.

        Returns:
            str: A formatted string containing the docstrings of all active commands.
        """
        if not self.commands:  # Should be initialized by initial_solve or constructor
            # Fallback if commands list is empty
            # This might happen if called before initial_solve populates it fully
            # For initial scaffold generation, only PaperReplace is active
            temp_commands_for_doc = [PaperReplace(self.save_loc)]
            if any(
                isinstance(cmd, PaperEdit) for cmd in self.commands
            ):  # Check if PaperEdit was added
                temp_commands_for_doc.append(PaperEdit(self.save_loc))  # type: ignore
            # Add Arxiv if it's intended to be available generally
            # temp_commands_for_doc.append(Arxiv())
        else:
            temp_commands_for_doc = self.commands  # type: ignore

        cmd_strings = "\n".join([cmd.docstring() for cmd in temp_commands_for_doc])
        return (
            "\nYou have access to tools which can be interacted with using the following structure: ```COMMAND_NAME\n<command arguments or content here>\n```\n"
            "Where COMMAND_NAME is the specific command (e.g., EDIT, REPLACE, SUMMARY, FULL_TEXT). The ``` delimiters MUST encapsulate the entire command block (name and arguments). "
            "YOU CAN ONLY EXECUTE A SINGLE COMMAND AT A TIME. Do not attempt multiple commands in one response.\n"
            f"{cmd_strings}"
        )

    def role_description(self) -> str:
        """Provides a description of the LLM's persona or role.

        Returns:
            str: A string describing the LLM's role (e.g., a PhD student).
        """
        return (
            "You are a computer science PhD student at a top university. You are preparing a paper for submission to a prestigious "
            "machine learning conference like ICLR. Your goal is to write an outstanding research paper that will be well-received by reviewers "
            "and accepted for publication. The paper should be approximately 8 pages (around 4000 words) and adhere to a standard "
            "scientific paper structure, typically including: 1. Abstract, 2. Introduction, 3. Related Work, 4. Background, "
            "5. Methods, 6. Experimental Setup, 7. Results, and 8. Discussion."
        )

    def phase_prompt(self) -> str:
        """Provides context for the current phase of the paper writing task.

        Returns:
            str: A string describing the current task objective for the LLM.
        """
        phase_str = (
            "You are currently focused on writing and refining a research paper. "
            "Your objective is to produce a high-quality, comprehensive, and coherent scientific document based on the provided research plan and topic. "
            "Pay close attention to the structure, clarity, technical depth, and correctness of the content. "
            "Use the available tools (commands) to edit, replace, or gather information as needed to improve the paper."
        )
        return phase_str
