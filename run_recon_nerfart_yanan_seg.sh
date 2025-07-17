


python launch.py --config configs/romap-recon.yaml --train --gpu 0 system.prompt_processor.prompt="A handsome man" system.seg_list=[['eyes'],['nose'],['lips'],['hair'],['face']] system.if_segment=True system.if_gen=False system.recon_path='./dataset/yanan/yanan.ply' system.if_recon=True data.camera_distance=1 data.eval_interpolation=[] data.camera_layout="front" data.dataroot="./dataset/yanan" system.dataroot="./dataset/yanan" system.seg_softmax_list=[0.1,0.1,0.1,0.2,0.7] data.rot_name="yanan" data.fov=0.6
