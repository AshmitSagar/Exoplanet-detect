# Contributing to AstroVet-ISRO

Thank you for your interest in contributing to **AstroVet-ISRO**! We welcome improvements, bug reports, astrophysics validation datasets, and documentation updates.

---

## Code of Conduct

Please maintain a collaborative, respectful, and scientifically rigorous environment.

---

## Getting Started

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/AshmitSagar/Exoplanet-detect.git
   cd Exoplanet-detect
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Test Suite**
   ```bash
   pytest -v -m "not network"
   ```

---

## Development Workflow

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write clean, modular, and documented Python code following PEP 8.
3. Ensure all tests pass and add unit tests in `tests/` for any new functionality.
4. Avoid committing large datasets, `.pth` model binaries, or raw FITS files directly into git history.

---

## Running Smoke Tests

Verify that your training setup and model backward pass work correctly:
```bash
python src/pipeline/AI/training/train.py --smoke_test
```

---

## Submitting a Pull Request

- Provide a clear PR title and description outlining what was changed and why.
- Reference any relevant issues or NASA/ISRO dataset specifications.
- Ensure continuous integration and pytest checks pass cleanly.
