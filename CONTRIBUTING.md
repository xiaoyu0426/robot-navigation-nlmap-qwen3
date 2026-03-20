# Contributing to NLMap + Qwen3 Robot Navigation System

Thank you for considering contributing to our project!

## How to Contribute

### Reporting Bugs
- Search existing issues first
- Open a new issue with clear description
- Include code samples and expected vs actual behavior

### Suggesting Enhancements
- Open a new issue with clear title and description
- Explain why this feature would be useful

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

### Development Setup
```bash
# Clone repository
git clone https://github.com/xiaoyu0426/robot-navigation-nlmap-qwen3.git
cd robot-navigation-nlmap-qwen3

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-4B models/Qwen3-models
```

### Code Style
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Comment code where necessary
