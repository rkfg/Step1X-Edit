python inference.py --input_dir ./examples \
    --model_path ~/.cache/huggingface/hub/models--meimeilook--Step1X-Edit-FP8/snapshots/fd7c58e1ab6283bbb7a832a62d8df9363ebc4f78 \
    --json_path ./examples/prompt_en.json \
    --output_dir ./output_en \
    --seed 1234 --size_level 1024 

python inference.py --input_dir ./examples \
    --model_path /data/work_dir/step1x-edit/ \
    --json_path ./examples/prompt_cn.json \
    --output_dir ./output_cn \
    --seed 1234 --size_level 1024 