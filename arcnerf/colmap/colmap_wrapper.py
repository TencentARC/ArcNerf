# -*- coding: utf-8 -*-

import os
import subprocess


def run_colmap(scene_dir, logger, match_type='exhaustive_matcher'):
    """Run colmap by command line wrapper. """
    logfile_name = os.path.join(scene_dir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    # feature extraction. Same camera models
    feature_extractor_args = [
        'colmap',
        'feature_extractor',
        '--database_path',
        os.path.join(scene_dir, 'database.db'),
        '--image_path',
        os.path.join(scene_dir, 'images'),
        '--ImageReader.single_camera',
        '1',
    ]
    feat_output = subprocess.check_output(feature_extractor_args, universal_newlines=True)
    logfile.write(feat_output)
    logger.add_log('    Features extracted...')

    # feature matching
    feature_matcher_args = [
        'colmap',
        match_type,
        '--database_path',
        os.path.join(scene_dir, 'database.db'),
    ]

    matcher_output = subprocess.check_output(feature_matcher_args, universal_newlines=True)
    logfile.write(matcher_output)
    logger.add_log('    Features matched...')

    # sparse mapping
    p = os.path.join(scene_dir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    mapper_args = [
        'colmap',
        'mapper',
        '--database_path',
        os.path.join(scene_dir, 'database.db'),
        '--image_path',
        os.path.join(scene_dir, 'images'),
        '--output_path',
        os.path.join(scene_dir, 'sparse'),
        # '--Mapper.num_threads', '16',
        # '--Mapper.init_min_tri_angle', '4',
        # '--Mapper.multiple_models', '0',
        # '--Mapper.extract_colors', '0',
    ]

    map_output = subprocess.check_output(mapper_args, universal_newlines=True)
    logfile.write(map_output)
    logger.add_log('    Sparse map created')

    logfile.close()
    logger.add_log('Finished running COLMAP, see {} for logs'.format(logfile_name))


def run_colmap_dense(scene_dir, logger):
    """Run dense reconstruction and produce mesh"""
    logfile_name = os.path.join(scene_dir, 'colmap_dense_output.txt')
    logfile = open(logfile_name, 'w')

    p = os.path.join(scene_dir, 'dense')
    if not os.path.exists(p):
        os.makedirs(p)

    # Image undistorer
    image_undistorter_args = [
        'colmap',
        'image_undistorter',
        '--image_path',
        os.path.join(scene_dir, 'images'),
        '--input_path',
        os.path.join(scene_dir, 'sparse/0'),
        '--output_path',
        os.path.join(scene_dir, 'dense'),
        '--output_type',
        'COLMAP',
    ]
    undistorter_output = subprocess.check_output(image_undistorter_args, universal_newlines=True)
    logfile.write(undistorter_output)
    logger.add_log('    Image distorter done...')

    # patch_match_stereo
    patch_path = os.path.join(scene_dir, 'dense/stereo')
    if not os.path.exists(patch_path):
        patch_matcher_args = [
            'colmap',
            'patch_match_stereo',
            '--workspace_path',
            os.path.join(scene_dir, 'dense'),
            '--PatchMatchStereo.geom_consistency',
            'true',
            '--workspace_format',
            'COLMAP',
        ]
        patch_matcher_output = subprocess.check_output(patch_matcher_args, universal_newlines=True)
        logfile.write(patch_matcher_output)
        logger.add_log('    Patch Matcher done...')

    # stereo_fusion
    fusion_path = os.path.join(scene_dir, 'dense/fused.ply')
    if not os.path.exists(fusion_path):
        stereo_fusion_args = [
            'colmap',
            'stereo_fusion',
            '--workspace_path',
            os.path.join(scene_dir, 'dense'),
            '--input_type',
            'geometric',
            '--workspace_format',
            'COLMAP',
            '--output_path',
            fusion_path,
        ]
        stereo_fusion_output = subprocess.check_output(stereo_fusion_args, universal_newlines=True)
        logfile.write(stereo_fusion_output)
        logger.add_log('    Stereo Fusion extracted at ' + fusion_path)

    # poisson_mesher
    poisson_path = os.path.join(scene_dir, 'dense/meshed-poisson.ply')
    if not os.path.exists(poisson_path):
        poisson_mesher_args = [
            'colmap',
            'poisson_mesher',
            '--input_path',
            os.path.join(scene_dir, 'dense'),
            '--output_path',
            poisson_path,
        ]
        poisson_mesher_output = subprocess.check_output(poisson_mesher_args, universal_newlines=True)
        logfile.write(poisson_mesher_output)
        logger.add_log('    Poisson mesher extracted at ' + poisson_path)

    logfile.close()
    logger.add_log('Finished running COLMAP dense reconstruct, see {} for logs'.format(logfile_name))
    logger.add_log('Mesh save at {}...'.format(os.path.join(scene_dir, 'dense/fused.ply')))
