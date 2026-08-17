# Taxi Fare Prediction Final Project

This MLOps final project uses an XGBoost model to predict NYC taxi trip fares from trip distance, passenger count, and pickup/dropoff location IDs. It covers the full MLOps lifecycle from experiment tracking and model registry, a FastAPI backend, a user interface, model monitoring, testing, CI/CD, containerization, and deployment. For the experiment tracking, I used Weights & Biases and AWS for deployment including DynamoDB and EC2 for storage and hosting.
---
## Setup Instructions
Clone the repo and set up a virtual environment:
'''
  git clone https://github.com/Nathaniel-Benton/Summer_2025_DU_MLOPS_Final_Project_Taxi_Fare.git
  cd Summer_2025_DU_MLOPS_Final_Project_Taxi_Fare
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
'''
---
Configure Weights & Biases:
'''
  wandb login
'''
---
Paste your Weights and Biases API key into the terminal after running that command.

If need be, you will need to train a new model to be tracked and logged within W&B automatically:
'''
  python3 training/train_model_xgb_v2.2.py
'''
---
Configure AWS credentials for running locally to test all applications. These will need to be updated each time the AWS console is started:
'''
~/.aws/credentials

  [default]
  aws_access_key_id=YOUR_ACCESS_KEY
  aws_secret_access_key=YOUR_SECRET_KEY
  aws_session_token=YOUR_SESSION_TOKEN
'''
'''
~/.aws/config:

  [default]
  region=us-east-1
'''
---
Setting up DynamoDB in the AWS Learner Lab
In the AWS Console, go to DynamoDB → Tables → Create table and name it taxi-fare-predictions. For the partition key it was set to prediction_id. Under Table settings confirm that the capacity is set to on-demand and leave all other defaults and click Create table. Make sure the table status showsActive before running any of the apps that connect to it.

No other setup is required — `main.py` and `monitoring_dashboard.py` both connect to this table automatically via `boto3`, as long as valid AWS credentials are available (see Setup Instructions above).

---
Run the app locally
Backend API
'''
  uvicorn main:app --reload --port 8001
'''
  Open in your browser using: http://127.0.0.1:8001/docs

Frontend User Interface
'''
  streamlit run streamlit_app.py 
'''
  Open in your browser using: http://localhost:8501

Monitoring Dashboard
'''
  streamlit run monitoring_dashboard.py --server.port 8502
'''
  dashboard, http://localhost:8502
---
Run tests
'''
  pytest tests/ -v
  ruff check .
'''

CI/CD has also been set up on GitHub — .github/workflows/ci.yml automatically runs these same two checks on every pull request into main, and a branch protection rule blocks merging if either one fails.

## Deployment Steps
To test the applications without the EC2 instances prior to setting up those, the following steps can be taken:

Open the AWS learner lab.

Update an .env file and fill in current AWS credentials and your W&B API key
  The .env file is set up with the AWS credentials and W&B API in all upper case, so when copying from the AWS Details tab in the AWS learner lab, you need to only copy the credentials. The format looks like this:

  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...
  AWS_SESSION_TOKEN=...
  AWS_DEFAULT_REGION=us-east-1
  WANDB_API_KEY=...
---

Docker builds for:
Backend and Frontend files:
'''
  docker compose up --build 
'''

Monitoring files:
'''
  docker build -f Dockerfile.monitoring -t taxi-fare-monitoring .
  docker run -p 8502:8502 --env-file .env taxi-fare-monitoring
'''

This will run all files on the local host and will allow health, predictions, and feeback submitals to be sent to the AWS DynamoDB data storage. This will communicate with the monitoring dashboard to allow for all monitoring checks.
---

Launching the EC2 instances

To launch all three EC2 instances, you will need to repeat these steps three times.

In the AWS Console, go to EC2 → Instances → Launch instance. Name the instance taxi-fare-backend, taxi-fare-frontend, and taxi-fare-monitoring. Each EC2 instance was set up with Amazon Linux 2023, with the t2.micro type. 

For the first instance set up, you will need to create a key pair using RSA .pem format. This will need to be downloaded and reused for all instances.

The network settings should be set to allow SSH traffic with My IP selected. As a second security group rule, it should be set to a Custom TCP with a port range of 8001 for the backend, 8501 for the frontend, and 8502 for the monitoring instance with the source set to Anywhere.Once this is set up, you will need to pressLaunch instance and wait for status to show Running. Next you will copy the Public IPv4 address to add to the ssh connection point.
---
'''
chmod 400 ~/.ssh/<your-key>.pem
ssh -i ~/.ssh/<your-key>.pem ec2-user@<INSTANCE_PUBLIC_IP>
'''

---
For setting this MLOPs up with the EC2 instances, the following steps should be taken after the AWS instances are set up on the AWS cloud:
AWS EC2 set up: the backend, prediction frontend, and monitoring dashboard each run on their own t3.micro EC2 instance (a total of three instances).

For each instance on the AWS server, the initial set up can be seen below:

The .pem key and and IP addresses will need to be updated based on the user and the instance IP.
'''
  ssh -i ~/.ssh/<your-key>.pem ec2-user@<INSTANCE_PUBLIC_IP>
  sudo yum update -y && sudo yum install -y docker git
  sudo service docker start
  sudo usermod -aG docker ec2-user
  exit
  ssh -i ~/.ssh/<your-key>.pem ec2-user@<INSTANCE_PUBLIC_IP>   # reconnect
  git clone https://github.com/Nathaniel-Benton/Summer_2025_DU_MLOPS_Final_Project_Taxi_Fare.git
  cd Summer_2025_DU_MLOPS_Final_Project_Taxi_Fare
'''
---

Backend and monitoring instances need AWS/W&B credentials — copy your local .env over rather than retyping it (long values like session tokens can get corrupted over SSH):

Backend:
'''
  scp -i ~/.ssh/<your-key>.pem .env ec2-user@<INSTANCE_PUBLIC_IP>:~/Summer_2025_DU_MLOPS_Final_Project_Taxi_Fare/.env
  docker build -f Dockerfile.backend -t taxi-fare-backend .
  docker run -d -p 8001:8001 --env-file .env --name backend taxi-fare-backend
'''

Monitoring:
'''
  scp -i ~/.ssh/<your-key>.pem .env ec2-user@<INSTANCE_PUBLIC_IP>:~/Summer_2025_DU_MLOPS_Final_Project_Taxi_Fare/.env
  docker build -f Dockerfile.monitoring -t taxi-fare-monitoring .
  docker run -d -p 8502:8502 --env-file .env --name monitoring taxi-fare-monitoring
'''

Frontend:
'''
  docker build -f Dockerfile.frontend -t taxi-fare-frontend .
  docker run -d -p 8501:8501 -e API_URL=http://<BACKEND_PUBLIC_IP>:8001 --name frontend taxi-fare-frontend
'''
---

## Example Requests by User

API health
'''
  curl http://<BACKEND_PUBLIC_IP>:8001/health
'''
This should print the message:
  {"status": "ok", "message": "API is running and model is loaded"}
---

API prediction
'''
  curl -X POST http://<BACKEND_PUBLIC_IP>:8001/predict \
   -H "Content-Type: application/json" \
    -d '{
      "trip_distance": 3.5,
      "passenger_count": 1,
      "PULocationID": 142,
     "DOLocationID": 236,
     "RatecodeID": 1
    }'
'''
This should pring a message something like this:
  {
    "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "predicted_fare_amount": 18.42
  }
---

API feedback
'''
  curl -X POST http://<BACKEND_PUBLIC_IP>:8001/feedback \
    -H "Content-Type: application/json" \
    -d '{
      "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
      "actual_fare": 19.00
    }'
'''
This should print something like this:
  {"status": "feedback recorded", "prediction_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}
---

Or, you can use the UI directly by launching it and interacting with it there either locally, or while set up on the instances. When doing so, you will be able to determine if it is working if the predictions are appearing in the DynamoDB database.