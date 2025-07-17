

# #! dino
python launch.py --config configs/romap-recon.yaml --train --gpu 0 system.prompt_processor.prompt="a photo of dinosaur figurine" system.seg_list=[['head'],['body']]  system.if_segment=True system.if_gen=False system.recon_path='./dataset/room/point_cloud/iteration_30000/point_cloud.ply' system.if_recon=True data.camera_distance=1 data.eval_interpolation=[0,1,2,50] data.camera_layout="front" data.dataroot="./dataset/room/dinosaur" system.dataroot="./dataset/room/dinosaur" system.seg_softmax_list=[0.2,0.2,0.2] data.rot_name="custom" data.fov=1.0

