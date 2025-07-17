

# test
python infer.py big --resume pretrained/model_fp16.safetensors --workspace workspace_test --test_path data_test

# # gradio app
# python app.py big --resume workspace/model.safetensors

# # local gui
# python gui.py big --output_size 800 --test_path workspace_test/anya_rgba.ply

# # mesh conversion
# python convert.py big --test_path workspace_test/anya_rgba.ply