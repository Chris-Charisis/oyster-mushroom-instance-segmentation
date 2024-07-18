# Oyster Mushroom Segmentation API

This repository contains a simplified API developed using Docker that deploys a trained Mask R-CNN model to segment individual oyster mushrooms from images of mushroom clusters. The authorization is implemented with a simple token hashing.

## Getting Started
### Prerequisites
- Docker
- Python 3.8+

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/oyster-mushroom-segmentation-api.git
cd oyster-mushroom-segmentation-api
```
2. **Build the Docker image:**
```bash
docker build -t oyster-mushroom-segmentation-api .
```
3. **Run the Docker container:**
```bash
docker run -p 8000:8000 oyster-mushroom-segmentation-api
```
### API Usage
Once the Docker container is running, the API will be available at http://localhost:8000. You can send a POST request with an image of oyster mushroom clusters to the /process_image endpoint. The API will return the image with the segmented individual mushrooms.

```bash
curl -X POST "http://localhost:8000/process_image/"  -H "Content-Type: application/x-www-form-urlencoded" -d "ADD_A_VALID_URL_TO_AN_IMAGE"  -d "token=mushnomics_ucd" --output ./output.jpg
```

