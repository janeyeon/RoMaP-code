python launch.py --config configs/romap-gen.yaml --train --gpu 0 system.prompt_processor.prompt="A face of beautiful woman" system.seg_list=[['lips'],['hair'],['face']] system.if_segment=True system.if_gen=True system.ply_path='./dataset/woman_2.ply' system.if_recon=False system.seg_softmax_list=[0.1,0.2,0.4]

