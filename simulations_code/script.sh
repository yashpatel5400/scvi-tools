python make_dataset.py --output-dir /home/ubuntu/simu_runs/run_A
python run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_A --sc-epochs 1 --st-epochs 1 --index-key 0
python run_stereoscope.py --input-dir /home/ubuntu/simu_runs/run_A --sc-epochs 1 --st-epochs 1 --index-key 1
python run_destVI.py --input-dir /home/ubuntu/simu_runs/run_A --sc-epochs 1 --st-epochs 1
python run_harmony.py --input-dir /home/ubuntu/simu_runs/run_A
Rscript --vanilla run_RCTD.R  /home/ubuntu/simu_runs/run_A /RCTD/


python eval_scvi-tools_model.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir destvi --model-string DestVI
python eval_scvi-tools_model.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir stereo2th_sub-cell_type --model-string Stereoscope1
python eval_scvi-tools_model.py --input-dir /home/ubuntu/simu_runs/run_A --model-subdir stereocell_type --model-string Stereoscope0