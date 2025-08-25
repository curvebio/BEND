#!/bin/bash

# BEND Development Setup Script
# This script provides an interactive setup for the BEND project

set -e  # Exit on any error

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧬 BEND - Benchmark of DNA Language Models${NC}"
echo -e "${BLUE}==========================================${NC}"
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed"
        print_info "Please install uv first: https://docs.astral.sh/uv/"
        print_info "Quick install: curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    print_status "uv is installed"
}

# Check if make is available
check_make() {
    if command -v make &> /dev/null; then
        print_status "make is available"
        return 0
    else
        print_warning "make is not available, using direct commands"
        return 1
    fi
}

# Setup environment using make or direct commands
setup_environment() {
    local use_make=$1
    
    print_info "Setting up Python 3.11 environment..."
    
    if [ "$use_make" = true ]; then
        make setup
    else
        # Direct setup without make
        print_info "Installing Python 3.11..."
        uv python install 3.11
        
        print_info "Creating virtual environment..."
        uv venv --python 3.11
        
        print_info "Installing dependencies..."
        uv pip install -r requirements.txt
        
        print_info "Installing BEND in development mode..."
        source .venv/bin/activate && uv pip install -e .
    fi
    
    print_status "Environment setup complete!"
}

# Setup development tools
setup_dev_tools() {
    local use_make=$1
    
    print_info "Setting up development tools..."
    
    if [ "$use_make" = true ]; then
        make install-dev
    else
        source .venv/bin/activate && uv pip install pytest pytest-cov black isort flake8 mypy pre-commit sphinx sphinx-rtd-theme myst-parser
    fi
    
    print_status "Development tools installed!"
}

# Check if gsutil is available
check_gsutil() {
    if command -v gsutil &> /dev/null; then
        print_status "gsutil is available"
        return 0
    else
        print_warning "gsutil is not available"
        print_info "Install Google Cloud SDK for faster downloads: https://cloud.google.com/sdk/docs/install"
        return 1
    fi
}

# Download data
download_data() {
    local use_make=$1
    
    read -p "Do you want to download the BEND dataset? (This may take a while) [y/N]: " download_choice
    if [[ $download_choice =~ ^[Yy]$ ]]; then
        print_info "Downloading BEND dataset..."
        
        if [ "$use_make" = true ]; then
            if check_gsutil; then
                print_info "Using fast Google Cloud Storage download..."
                make download-data
            else
                print_info "Falling back to original download script..."
                make download-data-original
            fi
        else
            if check_gsutil; then
                print_info "Using fast Google Cloud Storage download..."
                mkdir -p data
                gsutil -m cp -r gs://curvebio-mahdibaghbanzadeh/bend/* data/
            else
                print_info "Using original download script..."
                source .venv/bin/activate && python scripts/download_bend.py
            fi
        fi
        
        print_status "Dataset downloaded!"
    else
        print_info "Skipping dataset download."
        if check_gsutil; then
            print_info "You can download it later with: make download-data (fast)"
        fi
        print_info "Or use: make download-data-original (original method)"
    fi
}

# Main setup flow
main() {
    echo "This script will help you set up the BEND development environment."
    echo ""
    
    # Check prerequisites
    check_uv
    local has_make=false
    if check_make; then
        has_make=true
    fi
    
    echo ""
    echo "Setup options:"
    echo "1. Basic setup (required dependencies only)"
    echo "2. Development setup (includes testing and formatting tools)"
    echo "3. Full setup (development + dataset download)"
    echo ""
    
    read -p "Choose setup type [1-3]: " setup_choice
    
    case $setup_choice in
        1)
            setup_environment $has_make
            ;;
        2)
            setup_environment $has_make
            setup_dev_tools $has_make
            ;;
        3)
            setup_environment $has_make
            setup_dev_tools $has_make
            download_data $has_make
            ;;
        *)
            print_error "Invalid choice. Please run the script again."
            exit 1
            ;;
    esac
    
    echo ""
    print_status "Setup complete!"
    echo ""
    print_info "To activate the environment:"
    echo "  source .venv/bin/activate"
    echo ""
    
    if [ "$has_make" = true ]; then
        print_info "Available make commands:"
        echo "  make help           # Show all available commands"
        echo "  make status         # Check environment status"
        echo "  make test           # Run tests"
        echo "  make format         # Format code"
        echo "  make download-data  # Download dataset"
    else
        print_info "You can also install 'make' to use convenient shortcuts:"
        echo "  sudo apt-get install make  # On Ubuntu/Debian"
        echo "  brew install make         # On macOS"
    fi
    
    echo ""
    print_info "Next steps:"
    echo "1. Activate the environment: source .venv/bin/activate"
    echo "2. Explore the examples/ directory"
    echo "3. Read the documentation: https://bend.readthedocs.io"
    echo "4. Check out the Jupyter notebooks in examples/"
}

# Run main function
main "$@"
