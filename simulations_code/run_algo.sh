#!/bin/bash

python run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_A --sc-epochs 15 --st-epochs 2000 --index-key 0
python run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_A --sc-epochs 15 --st-epochs 2000 --index-key 1
python run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_A --sc-epochs 15 --st-epochs 2000 --index-key 2
python run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_A --sc-epochs 15 --st-epochs 2000 --index-key 3
python run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_A --sc-epochs 15 --st-epochs 2000 --index-key 4
python run_destVI.py --input-dir /home/ubuntu/simu_runs/run_A --sc-epochs 15 --st-epochs 2500
python run_embedding.py --input-dir /home/ubuntu/simu_runs/run_A --output-suffix harmony --algorithm Harmony
python run_embedding.py --input-dir /home/ubuntu/simu_runs/run_A --output-suffix scanorama --algorithm Scanorama
python run_embedding.py --input-dir /home/ubuntu/simu_runs/run_A --output-suffix scvi --algorithm scVI
Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_A /RCTD0/ 0
Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_A /RCTD1/ 1
Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_A /RCTD1/ 2
Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_A /RCTD1/ 3
Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_A /RCTD1/ 4