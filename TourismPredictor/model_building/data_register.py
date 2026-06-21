from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

from google.colab import userdata

# Get the HF_TOKEN from Colab secrets and set it as an environment variable
hf_token = userdata.get('HF_TOKEN')
os.environ["HF_TOKEN"] = hf_token

# Initialize HfApi with the token
api = HfApi(token=os.getenv("HF_TOKEN"))


repo_id = "Shaktidaran/TourismPredictor"

repo_type = "dataset"



# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="TourismPredictor/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
