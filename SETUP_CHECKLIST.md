# Phase 0 Setup Checklist

You drive these steps. I scaffolded the project; the items below are accounts and credentials that only you can create. Tick each box as you go. If anything fails, copy the exact error message and ask me for help.

Estimated time end to end: 60 to 90 minutes.

## 1. Local Python environment

- [ ] Confirm Python 3.11 is installed: `python3 --version`
- [ ] From the project root, create a virtual environment: `python3 -m venv forecast-env`
- [ ] Activate it: `source forecast-env/bin/activate` (Mac/Linux)
- [ ] Upgrade pip: `pip install --upgrade pip`
- [ ] Install all dependencies: `pip install -r requirements.txt`

Heads up: Prophet and LightGBM both pull in compiled wheels. On Mac with Apple Silicon you may need `brew install cmake libomp` first if LightGBM fails to build.

## 2. AWS account and IAM user

- [ ] Create or sign in at https://aws.amazon.com
- [ ] Open the IAM console
- [ ] Users -> Create user -> name it `ml-project-user`
- [ ] Attach the following managed policies (search and tick each):
  - `AmazonSageMakerFullAccess`
  - `AmazonS3FullAccess`
  - `AWSLambdaFullAccess`
  - `CloudWatchFullAccess`
  - `IAMReadOnlyAccess`
- [ ] After creation, open the user, go to Security credentials -> Create access key -> choose "Command Line Interface", download the CSV
- [ ] Install AWS CLI: `pip install awscli` (already in requirements.txt) or `brew install awscli`
- [ ] Run `aws configure` and paste the Access Key ID, Secret, region `eu-west-1`, output `json`
- [ ] Verify it works: `aws sts get-caller-identity` should return your account number

Important: never commit the CSV or `.aws/credentials` to git. The `.gitignore` already blocks common patterns but be vigilant.

## 3. SageMaker execution role

SageMaker training jobs run as a separate role, not as your IAM user. Create one:

- [ ] In IAM, Roles -> Create role -> Trusted entity type: AWS service -> Use case: SageMaker -> Next
- [ ] Attach `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`
- [ ] Name it `SageMakerExecutionRole`
- [ ] Copy the Role ARN (looks like `arn:aws:iam::123456789012:role/SageMakerExecutionRole`)
- [ ] Paste the ARN into your `.env` (copy `.env.example` to `.env` first)

## 4. S3 bucket

- [ ] Pick a globally unique bucket name. Replace `yourname` with something distinctive: `travel-forecast-yourname`
- [ ] Create it: `aws s3 mb s3://travel-forecast-yourname --region eu-west-1`
- [ ] Verify: `aws s3 ls`
- [ ] Update `.env` with the bucket name

## 5. Kaggle API key

- [ ] Sign in or create an account at https://www.kaggle.com
- [ ] Profile picture -> Settings -> scroll to API -> Create New Token
- [ ] This downloads `kaggle.json`
- [ ] Move it: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/`
- [ ] Lock the file: `chmod 600 ~/.kaggle/kaggle.json`
- [ ] Test: `kaggle datasets list -s hotel-booking-demand` should show the dataset

## 6. Pull the dataset

- [ ] From the project root: `kaggle datasets download -d jessemostipak/hotel-booking-demand -p data/raw`
- [ ] Unzip it: `cd data/raw && unzip hotel-booking-demand.zip && cd ../..`
- [ ] You should now have `data/raw/hotel_bookings.csv` (about 16 MB, 119,390 rows)

## 7. GitHub repository

- [ ] Create a new GitHub repo named `travel-demand-forecasting` (private is fine while in development)
- [ ] Do NOT add a README, gitignore, or license through the GitHub UI; we already have those locally
- [ ] In the project folder run:

```
git init
git add .
git commit -m "Initial scaffold for travel demand forecasting platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/travel-demand-forecasting.git
git push -u origin main
```

- [ ] Verify the push worked by reloading the GitHub page

## 8. Final smoke test

- [ ] In your activated venv, run: `python -c "import pandas, prophet, lightgbm, sagemaker, boto3; print('all imports ok')"`
- [ ] Run: `python -c "import pandas as pd; df=pd.read_csv('data/raw/hotel_bookings.csv'); print(df.shape)"` should print `(119390, 32)`
- [ ] Run `aws sts get-caller-identity` and confirm you see your account number

If all three pass, Phase 0 is done. Move on to Phase 1 (EDA) by opening `notebooks/01_eda.ipynb`.

## Common gotchas

- Prophet wheel install fails on Mac M1/M2: run `brew install cmake libomp` then retry pip install
- AWS CLI says "Unable to locate credentials": you skipped `aws configure` or used the wrong profile
- Kaggle 403: your `~/.kaggle/kaggle.json` is missing or has wrong permissions
- `git push` rejects with "non fast forward": you accidentally created the repo with files on GitHub. Run `git pull --rebase origin main` then push again

When you finish or get stuck on any step, ping me and I'll help debug.
