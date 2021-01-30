#python painter_gmcnn.py --load_model_dir ./checkpoints/blind_face_stroke/ --img_shapes 256,256 --mode silent
python painter_gmcnn.py --model vcn_lite --load_model_dir ../focalnet_checkpoints/celebahq-blind-58-100/ --img_shapes 256,256 --mode silent
python painter_gmcnn.py --model vcn_lite --load_model_dir ../focalnet_checkpoints/ffhq-vcn_lite/ --img_shapes 256,256 --mode silent
python painter_gmcnn.py --model vcn_lite --load_model_dir ../focalnet_checkpoints/places2full-vcn/ --img_shapes 512,680 --mode silent
python painter_gmcnn.py --model vcn_lite --load_model_dir ../focalnet_checkpoints/paris-vcn_lite/ --img_shapes 256,256 --mode silent
