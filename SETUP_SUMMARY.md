# BEND Environment Setup Summary

This document summarizes the streamlined environment setup for the BEND project.

## What Was Updated

### 1. pyproject.toml
- **Modernized Python packaging**: Replaced the old setup.py approach with a comprehensive pyproject.toml
- **Build system configuration**: Added proper build system requirements
- **Project metadata**: Complete project information, authors, keywords, and classifiers
- **Dependencies management**: Moved from requirements.txt to pyproject.toml (while keeping requirements.txt for compatibility)
- **Development dependencies**: Added optional dev dependencies for testing, linting, and documentation
- **Tool configurations**: Added configurations for black, isort, pytest, mypy, and coverage
- **Python version constraint**: Set to Python 3.8-3.11 (3.12 excluded due to pathtools compatibility)
- **UV configuration**: Added uv-specific build dependencies to handle the pathtools/imp issue

### 2. Makefile
- **Comprehensive automation**: 20+ commands for all common development tasks
- **Environment management**: Setup, cleanup, and status checking
- **Development workflow**: Testing, linting, formatting, and documentation
- **Data management**: Dataset download and preprocessing
- **User-friendly help**: Color-coded output and comprehensive help system
- **Error handling**: Proper validation and informative error messages

### 3. Interactive Setup Script (setup_dev.sh)
- **User-friendly installation**: Interactive script for different setup scenarios
- **Flexible options**: Basic, development, or full setup
- **Fallback support**: Works even without make installed
- **Clear guidance**: Step-by-step instructions and next steps

### 5. Flexibility
- **Modern setup instructions**: Updated README.md with new quick setup process
- **Multiple installation methods**: Both make-based and manual installation
- **Clear prerequisites**: Emphasis on uv and Python 3.11 requirements

## Quick Setup Commands

### For new users:
```bash
# Clone and setup in one go
git clone https://github.com/frederikkemarin/BEND.git
cd BEND
make setup
source .venv/bin/activate
```

### For developers:
```bash
make setup-dev        # Install development tools
make format          # Format code
make test            # Run tests
make check           # Run all checks
```

### For data scientists:
```bash
make setup           # Basic setup
make download-data   # Get dataset via Google Cloud Storage (fast)
make run-notebook    # Start Jupyter
```

## Key Features

### 1. Fast Data Download
- **Google Cloud Storage**: Uses `gsutil` for efficient data transfer from `gs://curvebio-mahdibaghbanzadeh/bend`
- **Fallback support**: Original download script available as backup
- **Smart detection**: Automatically chooses best download method
- **Progress monitoring**: Enhanced status reporting with file counts and sizes

### 2. Python 3.11 Compatibility
- **Solved pathtools issue**: Uses Python 3.11 to avoid the imp module deprecation in 3.12
- **UV package manager**: Fast, modern Python package management
- **Virtual environment**: Isolated environment setup with .venv

### 2. Development Tools Integration
- **Code formatting**: Black and isort for consistent code style
- **Linting**: Flake8 for code quality checks
- **Type checking**: MyPy for static type analysis
- **Testing**: Pytest with coverage reporting
- **Documentation**: Sphinx with RTD theme

### 3. Streamlined Workflow
- **One-command setup**: `make setup` handles everything
- **Status monitoring**: `make status` shows environment health
- **Easy cleanup**: `make clean` and `make clean-env` for maintenance
- **Help system**: `make help` shows all available commands

### 4. Data Download Options
- **Fast download**: `make download-data` uses Google Cloud Storage (gsutil)
- **Fallback method**: `make download-data-original` uses original download script
- **Standalone script**: `./scripts/download_bend_gsutil.sh` for independent use
- **Smart detection**: Automatically detects available tools and chooses best method
- **Multiple setup methods**: Make, script, or manual installation
- **Modular installation**: Choose basic, dev, or full setup
- **Fallback compatibility**: Works even without make installed

## Migration Guide

If you have an existing BEND installation:

1. **Backup your work**: Commit any local changes
2. **Clean old environment**: `make clean-env` (or remove .venv manually)
3. **Fresh setup**: `make setup` or `./setup_dev.sh`
4. **Verify installation**: `make status`

## Troubleshooting

### Common Issues:
- **"uv not found"**: Install uv first: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **"make not found"**: Use `./setup_dev.sh` instead, or install make
- **"gsutil not found"**: Install Google Cloud SDK: `curl https://sdk.cloud.google.com | bash`
- **Download fails**: Use fallback method: `make download-data-original`
- **Python 3.12 issues**: The setup automatically uses Python 3.11 to avoid compatibility issues
- **Permission errors**: Make sure scripts are executable: `chmod +x setup_dev.sh scripts/download_bend_gsutil.sh`

### Getting Help:
- **Environment status**: `make status`
- **Available commands**: `make help`
- **Interactive setup**: `./setup_dev.sh`

## Future Maintenance

The new setup system makes it easy to:
- **Update dependencies**: Edit pyproject.toml and run `make setup`
- **Add new tools**: Update the dev dependencies and Makefile
- **Maintain consistency**: Use `make format` and `make check` regularly
- **Clean environments**: Use `make reset` for fresh installations

This modernized setup provides a robust, user-friendly foundation for BEND development and research.
