pip install openmim
mim install mmcv-full==1.6.0
python setup.py develop

# cpu too slow
python demo/restoration_video_demo.py  --device cuda --precision float32 --num_iter 3 --num_warmup 0 --profile
