#!/bin/bash

POSITIONAL_ARGS=()

FEATURES=false
TREE=false
MAPPER=false
REGISTER=false
while [[ $# -gt 0 ]]; do
    case $1 in
      --features)
        FEATURES=true
        shift
        ;;
      --tree)
        TREE=true
        shift
        ;;
      --mapper)
        MAPPER=true
        shift
        ;;
      --register)
        REGISTER=true
        shift
        ;;
      *)
        POSITIONAL_ARGS+=($1)
        shift
        ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

SAVE_DIR=$1
echo $SAVE_DIR
DATASET_DIR=$2
echo $DATASET_DIR
SCENE_INFO_DIR=$3
echo $SCENE_INFO_DIR
FILE=$4 
echo $FILE
NUM_THREADS=200
echo $NUM_THREADS
VOCAB_PATH=$5
echo $VOCAB_PATH

if [ ! -f "$FILE" ]; then
    [ ! -d "$SAVE_DIR"/sparse/reg ] && mkdir "$SAVE_DIR"/sparse/reg
    if [ "$FEATURES" = true ]; then
        colmap feature_extractor \
        --image_path "$DATASET_DIR" \
        --image_list_path "$SCENE_INFO_DIR"/register_frame_names_1fps.txt \
        --database_path "$SAVE_DIR"/database.db \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model RADIAL_FISHEYE \
        --ImageReader.camera_params "$(cat "$SCENE_INFO_DIR"/fisheye_intrinsics.txt)" \
        --SiftExtraction.gpu_index=1,6
    fi
    if [ "$TREE" = true ];then
        colmap vocab_tree_matcher \
        --database_path "$SAVE_DIR"/database.db \
        --VocabTreeMatching.vocab_tree_path  "${VOCAB_PATH}" \
        --VocabTreeMatching.match_list_path "$SCENE_INFO_DIR"/register_frame_names_1fps.txt \
        --SiftMatching.gpu_index=1,6
    fi
    if [ "$MAPPER" = true ];then
        colmap mapper \
        --database_path "$SAVE_DIR"/database.db \
        --image_path "$DATASET_DIR" \
        --image_list_path "$SCENE_INFO_DIR"/register_frame_names_1fps.txt \
        --output_path "$SAVE_DIR"/sparse

    fi
    if [ "$REGISTER" = true ];then
        echo "Extracting features of new images"
        colmap feature_extractor \
        --image_path "$DATASET_DIR" \
        --image_list_path "$SCENE_INFO_DIR"/register_frame_names_5fps.txt \
        --database_path "$SAVE_DIR"/database.db \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model RADIAL_FISHEYE \
        --ImageReader.camera_params "$(cat "$SCENE_INFO_DIR"/fisheye_intrinsics.txt)" \
        --SiftExtraction.gpu_index=1,6 

        echo "Matching images"
        colmap vocab_tree_matcher \
        --database_path "$SAVE_DIR"/database.db \
        --VocabTreeMatching.vocab_tree_path  "${VOCAB_PATH}" \
        --VocabTreeMatching.match_list_path "$SCENE_INFO_DIR"/register_frame_names_5fps.txt \
        --SiftMatching.gpu_index=1,6

        echo "Registering new images"
        colmap image_registrator \
        --database_path "$SAVE_DIR"/database.db \
        --input_path "$SAVE_DIR"/sparse/0 \
        --output_path "$SAVE_DIR"/sparse/reg \
        --Mapper.ba_refine_focal_length 0 \
        --Mapper.ba_refine_extra_params 0 \

        echo "Performing bundle_adjuster "
        colmap bundle_adjuster \
        --input_path "$SAVE_DIR"/sparse/reg \
        --output_path "$SAVE_DIR"/sparse/reg \
        --BundleAdjustment.refine_focal_length 0 \
        --BundleAdjustment.refine_extra_params  0

        echo "converting the model"
        colmap model_converter \
        --input_path "$SAVE_DIR"/sparse/reg \
        --output_path "$SAVE_DIR"/sparse/reg \
        --output_type TXT
        
        echo "Converting model pyl"
        colmap model_converter \
        --input_path "$SAVE_DIR"/sparse/reg \
        --output_path "$SAVE_DIR"/sparse/reg/sparse.ply \
        --output_type PLY
    fi
else
 echo "Found $FILE. Skipping image registration."
fi