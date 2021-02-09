# Step 1: make dataset
python make_dataset.py --output-dir /home/ubuntu/simu_runs/run_A

# Step 2: run algorithms
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


# Step 3: eval methods
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir destvi --model-string DestVI
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir stereo0 --model-string Stereoscope0
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir stereo1 --model-string Stereoscope1
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir stereo2 --model-string Stereoscope2
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir stereo3 --model-string Stereoscope3
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir stereo4 --model-string Stereoscope4
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir harmony --model-string Harmony
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir scanorama --model-string Scanorama
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir scvi --model-string scVI
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir RCTD0 --model-string RCTD0
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir RCTD1 --model-string RCTD1
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir RCTD2 --model-string RCTD2
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir RCTD3 --model-string RCTD3
python -u benchmark.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir RCTD4 --model-string RCTD4
