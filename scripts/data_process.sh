# write video into cfgs.dir.data_dir/Capture/video_name/images. Controls video/image downsample
python tools/extract_video.py --configs configs/default.yaml \
--data.scene_name qq_tiger \
--data.video_path path_to_video \
--data.video_downsample 15 \
--data.image_downsample 4

# run colmap to extract poses from frames
python tools/extract_video.py --configs configs/default.yaml \
--data.scene_name qq_tiger \
--data.colmap.match_type 'sequential_matcher' \
--data.colmap.dense_reconstruct False
