

#! shampoo
python launch.py --config configs/romap-recon.yaml --train --gpu 0 system.prompt_processor.prompt="a photo of pump bottle" system.seg_list=[['pump','head'],['body']]  system.if_segment=True system.if_gen=False system.recon_path='./dataset/covered_desk/point_cloud/iteration_30000/point_cloud.ply' system.if_recon=True data.camera_distance=1 data.eval_interpolation=[0,1,2,50] data.camera_layout="front" data.dataroot="./dataset/covered_desk/shampoo" system.dataroot="./dataset/covered_desk/shampoo" system.seg_softmax_list=[0.2,0.2,0.2] data.rot_name="3d_ovs" data.fov=1.0

