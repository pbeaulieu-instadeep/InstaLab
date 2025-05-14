# InstaLab

InstaLab is a multi-agent system for automated scientific research.
It simulates a laboratory environment with different agents (PhD Student, Postdoc, Professor, Reviewers)
collaborating to perform research tasks, from literature review to report writing and refinement.

## Features

- **Multi-agent system** with specialized roles
- **Automated literature review** via arXiv
- **Research plan formulation**
- **Results interpretation**
- **LaTeX report writing**
- **Reviewer evaluation**
- **Multilingual support**
- **Human-in-the-loop** interface for supervision

## Installation

```bash
uv pip install -e
```

## Dependencies

- Python >= 3.12
- arxiv >= 2.2.0
- google-api-python-client >= 2.169.0
- google-generativeai >= 0.8.5
- pypdf >= 5.5.0
- pyyaml >= 6.0.2

## Configuration

1. Get a Google Gemini API key
2. Configure the API key via:
   - Environment variable: `GEMINI_API_KEY`
   - Or YAML configuration file

## Usage

Modify the config file and run:

```bash
ai_lab_repo --yaml-location "experiment_configs/BIO_agentrxiv.yaml"
```

## Project Structure

- `agents.py`: Definition of different agents (PhDStudent, Postdoc, Professor, Reviewers)
- `ai_lab_repo.py`: Main research workflow implementation
- `inference.py`: Gemini API interface
- `papersolver.py`: Document generation and modification management
- `tools.py`: Utility tools (arXiv search, etc.)
- `utils.py`: Various utility functions

## Research Phases

1. **Literature Review**: Search and analysis of academic papers
2. **Plan Formulation**: Research plan development
3. **Results Interpretation**: Analysis of experimental results
4. **Report Writing**: LaTeX report generation
5. **Report Refinement**: Improvement based on reviewer feedback

## Contributing

Contributions are welcome! Feel free to:

1. Fork the project
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

None

## Contact

p.beaulieu@instadeep.com
```
