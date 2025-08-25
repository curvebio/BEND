#!/bin/bash

# BEND Data Download Script
# Downloads BEND dataset from Google Cloud Storage

set -e  # Exit on any error

# Colors for output
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

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

echo -e "${BLUE}🧬 BEND Dataset Download${NC}"
echo -e "${BLUE}========================${NC}"
echo ""

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    print_error "gsutil is not installed"
    print_info "Please install Google Cloud SDK first:"
    print_info "https://cloud.google.com/sdk/docs/install"
    echo ""
    print_info "Quick install:"
    print_info "curl https://sdk.cloud.google.com | bash"
    print_info "exec -l \$SHELL"
    echo ""
    print_warning "Falling back to original download script..."
    if [ -f "scripts/download_bend.py" ]; then
        print_info "Running original download script..."
        if [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate && python scripts/download_bend.py
        else
            python scripts/download_bend.py
        fi
    else
        print_error "No fallback download method available"
        exit 1
    fi
    exit 0
fi

print_status "gsutil is available"

# Check if data directory exists
if [ -d "data" ] && [ "$(ls -A data)" ]; then
    print_warning "Data directory already exists and is not empty"
    read -p "Do you want to overwrite existing data? [y/N]: " overwrite_choice
    if [[ ! $overwrite_choice =~ ^[Yy]$ ]]; then
        print_info "Download cancelled"
        exit 0
    fi
    print_info "Removing existing data..."
    rm -rf data/*
fi

# Create data directory
mkdir -p data

# Show download information
echo ""
print_info "Download details:"
echo "  Source: gs://curvebio-mahdibaghbanzadeh/bend"
echo "  Destination: ./data/"
echo "  Method: Google Cloud Storage (gsutil)"
echo ""

# Estimate download size (optional, requires gsutil du)
print_info "Checking dataset size..."
DATASET_SIZE=$(gsutil du -sh gs://curvebio-mahdibaghbanzadeh/bend 2>/dev/null | awk '{print $1}' || echo "unknown")
if [ "$DATASET_SIZE" != "unknown" ]; then
    print_info "Dataset size: $DATASET_SIZE"
else
    print_warning "Could not determine dataset size"
fi

# Confirm download
read -p "Proceed with download? [Y/n]: " proceed_choice
if [[ $proceed_choice =~ ^[Nn]$ ]]; then
    print_info "Download cancelled"
    exit 0
fi

# Download the data
print_info "Starting download..."
echo ""

# Use gsutil with multiple threads for faster download
if gsutil -m cp -r gs://curvebio-mahdibaghbanzadeh/bend/* data/; then
    echo ""
    print_status "Download completed successfully!"
    
    # Show summary
    echo ""
    print_info "Download summary:"
    BED_FILES=$(find data -name "*.bed" | wc -l)
    HDF5_FILES=$(find data -name "*.hdf5" | wc -l)
    TOTAL_SIZE=$(du -sh data 2>/dev/null | cut -f1 || echo "unknown")
    
    echo "  BED files: $BED_FILES"
    echo "  HDF5 files: $HDF5_FILES"
    echo "  Total size: $TOTAL_SIZE"
    echo ""
    
    print_status "BEND dataset is ready for use!"
    print_info "Data location: ./data/"
    
else
    echo ""
    print_error "Download failed!"
    print_info "You can try:"
    print_info "1. Check your internet connection"
    print_info "2. Verify gsutil authentication: gsutil config"
    print_info "3. Use the fallback method: make download-data-original"
    exit 1
fi
