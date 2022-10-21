pip install openmim
mim install mmcv-full==1.6.0	# cuda
mim install mmcv-full==1.3.13   # atsm
python setup.py develop

# cpu too slow
# must float32
python demo/restoration_video_demo.py  --device cuda --precision float32 --num_iter 3 --num_warmup 0 --profile

python demo/restoration_video_demo.py  --device xpu --precision float32 --num_iter 3 --num_warmup 0
