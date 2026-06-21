# Nebius Compute Setup Notes

## Current state

- CLI profile `hackathon` is authenticated as service account `serviceaccount-e00twjcbm9f80s4pa6`.
- The service account's parent is an **AI project**: `aiproject-e00phab1pxk1ejwgmt`.
- `nebius compute instance create/list` reject `aiproject-...` as a parent because Compute resources must live under a **folder** or an **IAM project** (`project-...`), not an AI project.

## What you need from the Nebius console

1. Open https://console.nebius.com
2. Navigate to the tenant/folder that contains `aiproject-e00phab1pxk1ejwgmt`.
3. Copy the **folder ID** or **IAM project ID** (format `folder-...` or `project-...`).

## Option A: set the folder as the default parent for the profile

```bash
nebius --profile hackathon profile set --parent-id <folder-id>
```

After this, all compute commands will use that folder by default.

## Option B: pass parent-id on every command

```bash
nebius --profile hackathon compute instance list --parent-id <folder-id>
```

## Create a GPU VM for training

```bash
FOLDER_ID=<folder-id>
nebius --profile hackathon compute instance create \
  --parent-id "$FOLDER_ID" \
  --name g1-world-model \
  --platform gpu-h100-sxm \
  --preset 1gpu-h100-sxm \
  --zone eu-north1-c \
  --image-family ubuntu-22-04-lts \
  --ssh-user ubuntu \
  --ssh-public-key-file ~/.ssh/id_rsa.pub
```

If that preset is unavailable, try `gpu-a100-sxm` / `1gpu-a100-sxm` or a CPU preset like `cpu-e2`.

## Deploy and run training

After the VM is running:

```bash
VM_IP=<public-ip>
ssh ubuntu@$VM_IP

# On the VM:
git clone https://github.com/CodeWithInferno/calhacks.git
cd calhacks
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy or generate data, then train
python src/ingest_real_data.py --input data/raw_g1_data.csv --output data/g1_slope_load_data.csv
python src/train_world_model.py
```

## Service account permissions

If you still get permission errors, make sure the service account is added to the **editors** group for that folder/project in the IAM console.
