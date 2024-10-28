#local runtime
pip install virtualenv
brew install python@3.9
python3.9 -m venv .
gcloud config set project bagchi-genai-bb
gcloud auth application-default login
gcloud auth application-default set-quota-project bagchi-genai-bb
pip install cmake
pip install -r requirements.txt
  
streamlit run home.py \
  --browser.serverAddress=localhost \
  --server.enableCORS=false \
  --server.enableXsrfProtection=false \
  --server.port 8080



gcloud builds submit --tag gcr.io/bagchi-genai-bb/gaming-3d-assset
gcloud run deploy finance-advisor-app --image gcr.io/bagchi-genai-bb/gaming-3d-assset --platform managed  --allow-unauthenticated  --region us-central1



sudo ls -l /usr/local/bin | grep '../Library/Frameworks/Python*' | awk '{print $9}' | tr -d @ | xargs rm -f

sudo -H python3 -m ensurepip


# Embedding Generation
python3 images-to-embedding.py 
gsutil cp indexData.json gs://bagchi-genai-bb/images/indexData.json


gcloud ai indexes create \
  --metadata-file=index-metadata.json \
  --display-name=GamesDemo-MultiModal-Embeddings \
  --project=bagchi-genai-bb \
  --region=us-central1